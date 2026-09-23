//! 3D visualization of the binary system, rendered as sky-plane projections.
//!
//! This is a port of the `visualise` program from the original cpp-lcurve:
//! grids of points representing the two stars, the disc, its edges and the
//! gas stream are projected onto the plane of the sky at a given orbital
//! phase and rasterized as coloured dots. Still frames are written as PNG,
//! animations as GIF (pure Rust) or MP4 (via an external `ffmpeg` binary).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use roche::errors::RocheError;
use roche::{self, Etype, Point, RocheContext, Star, Vec3};

use crate::model::Model;
use crate::set_star_grid::star_eclipse;

/// Palette indices used by the rasterizer. These mirror the PGPLOT colour
/// indices of the original `visualise` program.
const BACKGROUND: u8 = 0;
const EDGES: u8 = 1;
const STAR2: u8 = 2;
const DISC: u8 = 3;
const STAR1: u8 = 4;
const NCOLOURS: usize = 5;

/// Rendering options shared by still frames and animations.
#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    pub width: usize,
    pub height: usize,
    pub x1: f64,
    pub x2: f64,
    pub y1: f64,
    pub y2: f64,
    pub reverse: bool,
    pub sdob: bool,
}

impl RenderOptions {
    /// Build validated options. When `height` is `None` it is derived from
    /// the aspect ratio of the plot limits; `force_even` rounds both
    /// dimensions down to even numbers (required by yuv420p MP4 encoding).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        height: Option<usize>,
        x1: f64,
        x2: f64,
        y1: f64,
        y2: f64,
        reverse: bool,
        sdob: bool,
        force_even: bool,
    ) -> Result<Self, RocheError> {
        if !(x2 > x1 && y2 > y1) {
            return Err(RocheError::ParameterError(
                "require x2 > x1 and y2 > y1.".to_string(),
            ));
        }
        let mut width = width;
        let mut height =
            height.unwrap_or_else(|| (width as f64 * (y2 - y1) / (x2 - x1)).round() as usize);
        if force_even {
            width &= !1;
            height &= !1;
        }
        if !(2..=16384).contains(&width) || !(2..=16384).contains(&height) {
            return Err(RocheError::ParameterError(
                "width and height must be between 2 and 16384 pixels.".to_string(),
            ));
        }
        Ok(Self {
            width,
            height,
            x1,
            x2,
            y1,
            y2,
            reverse,
            sdob,
        })
    }

    /// RGB palette matching the original PGPLOT colour scheme. `reverse`
    /// selects a white background with dark colours (the original default),
    /// otherwise a black background with bright colours. `sdob` swaps the
    /// colours of the two stars.
    pub fn palette(&self) -> [[u8; 3]; NCOLOURS] {
        let (bg, fg) = if self.reverse {
            ([255, 255, 255], [0, 0, 0])
        } else {
            ([0, 0, 0], [255, 255, 255])
        };
        let (red, green, blue) = if self.reverse {
            ([102, 0, 0], [0, 77, 0], [0, 0, 128])
        } else {
            ([255, 0, 0], [0, 255, 0], [0, 0, 255])
        };
        let (star2, star1) = if self.sdob { (blue, red) } else { (red, blue) };
        [bg, fg, star2, green, star1]
    }
}

/// The geometry of every plotted component, built once per render call and
/// shared (immutably) between frame-rendering threads.
pub struct Scene<'a> {
    star1: &'a [Point],
    star2: &'a [Point],
    disc: &'a [Point],
    outer_edge: Vec<Point>,
    inner_edge: Vec<Point>,
    stream: Vec<Point>,
    add_disc: bool,
    add_spot: bool,
    cofm: Vec3,
    iangle: f64,
}

/// Build the scene from a model and its pre-computed star and disc grids.
/// The disc edges are rebuilt densely for display (the flux grids are too
/// coarse to read as clean rim lines) and the gas stream is integrated with
/// `strinit`/`stradv`, exactly as the original `visualise` program does.
pub fn build_scene<'a>(
    model: &Model,
    star1: &'a [Point],
    star2: &'a [Point],
    disc: &'a [Point],
) -> Result<Scene<'a>, RocheError> {
    let (outer_edge, inner_edge) = if model.add_disc {
        (
            build_visual_disc_edge(model, true)?,
            build_visual_disc_edge(model, false)?,
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let stream = if model.add_spot {
        build_stream(model)?
    } else {
        Vec::new()
    };

    Ok(Scene {
        star1,
        star2,
        disc,
        outer_edge,
        inner_edge,
        stream,
        add_disc: model.add_disc,
        add_spot: model.add_spot,
        cofm: Vec3::new(model.q.value / (1.0 + model.q.value), 0.0, 0.0),
        iangle: model.iangle.value,
    })
}

/// Build a dense ring of points tracing the outer or inner disc edge for
/// display. The geometry and eclipse handling mirror `set_disc_edge_grid`
/// (top rim always visible unless eclipsed, wall visible from its own
/// side), but the sampling is chosen for image quality rather than flux
/// integration so the rim renders as a clean line.
fn build_visual_disc_edge(model: &Model, outer: bool) -> Result<Vec<Point>, RocheError> {
    const EFAC: f64 = 1.0000001;
    const NZ: usize = 5;

    let q: f64 = model.q.value;
    let iangle: f64 = model.iangle.value;

    let roche_context1 = RocheContext::new(q, Star::Primary, model.spin1.value)?;
    let roche_context2 = RocheContext::new(q, Star::Secondary, model.spin2.value)?;

    let (mut r1, mut r2) = model.get_r1r2();
    let rl1: f64 = roche_context1.x_l1;
    let rl2: f64 = 1.0 - roche_context2.x_l1;
    if r1 < 0.0 {
        r1 = rl1;
    }
    if r2 < 0.0 {
        r2 = rl2;
    }
    let ffac1: f64 = r1 / rl1;
    let ffac2: f64 = r2 / rl2;

    let rdisc1: f64 = if model.rdisc1.value > 0.0 {
        model.rdisc1.value
    } else {
        r1
    };
    let rdisc2: f64 = if model.rdisc2.value > 0.0 {
        model.rdisc2.value
    } else {
        model.radius_spot.value
    };

    let rad: f64 = if outer { rdisc2 } else { rdisc1 };
    let h: f64 = model.height_disc.value * rad.powf(model.beta_disc.value);

    // Sample the larger circle at 1/4-degree steps and scale the smaller
    // one with its circumference, so both rims render equally dense.
    let ntheta: usize = ((1440.0 * rad / rdisc2.max(rdisc1)) as usize).max(180);

    let add_eclipses = |posn: &Vec3| -> Result<Etype, RocheError> {
        let mut eclipses = Etype::new();
        if model.opaque {
            for eclipse_pair in roche::disc_eclipse(
                iangle,
                rdisc1,
                rdisc2,
                model.beta_disc.value,
                model.height_disc.value,
                posn,
            )? {
                eclipses.push(eclipse_pair);
            }
        }
        if model.eclipse1 {
            star_eclipse(
                &roche_context1,
                r1,
                ffac1,
                iangle,
                posn,
                model.delta_phase,
                model.roche1,
                Star::Primary,
                &mut eclipses,
            )?;
        }
        if model.eclipse2 {
            star_eclipse(
                &roche_context2,
                r2,
                ffac2,
                iangle,
                posn,
                model.delta_phase,
                model.roche2,
                Star::Secondary,
                &mut eclipses,
            )?;
        }
        Ok(eclipses)
    };

    let always_visible = Vec3::new(0.0, 0.0, 0.0);
    let columns: Vec<Vec<Point>> = (0..ntheta)
        .into_par_iter()
        .map(|i| -> Result<Vec<Point>, RocheError> {
            let theta: f64 = std::f64::consts::TAU * i as f64 / ntheta as f64;
            let (sint, cost) = theta.sin_cos();
            let (r_xy, wall_dirn) = if outer {
                (EFAC * rad, Vec3::new(cost, sint, 0.0))
            } else {
                (rad / EFAC, Vec3::new(-cost, -sint, 0.0))
            };

            let mut column = Vec::with_capacity(NZ + 1);

            // Top rim, visible from every direction unless eclipsed
            let posn = Vec3::new(r_xy * cost, r_xy * sint, EFAC * h);
            column.push(Point::new(
                posn,
                always_visible,
                0.0,
                1.0,
                add_eclipses(&posn)?,
            ));

            // Wall, visible from its own side
            for j in 0..NZ {
                let z: f64 = -h + 2.0 * h * j as f64 / (NZ - 1) as f64;
                let posn = Vec3::new(r_xy * cost, r_xy * sint, z);
                column.push(Point::new(posn, wall_dirn, 0.0, 1.0, add_eclipses(&posn)?));
            }
            Ok(column)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(columns.into_iter().flatten().collect())
}

/// Integrate the gas stream from the L1 point down to the bright-spot
/// radius, recording the phases at which each point is eclipsed by either
/// star (and the disc, if present).
fn build_stream(model: &Model) -> Result<Vec<Point>, RocheError> {
    let q: f64 = model.q.value;
    let iangle: f64 = model.iangle.value;

    let roche_context1 = RocheContext::new(q, Star::Primary, model.spin1.value)?;
    let roche_context2 = RocheContext::new(q, Star::Secondary, model.spin2.value)?;

    let (mut r1, mut r2) = model.get_r1r2();
    let rl1: f64 = roche_context1.x_l1;
    let rl2: f64 = 1.0 - roche_context2.x_l1;
    if r1 <= 0.0 {
        r1 = 0.99999999999 * rl1;
    }
    if r2 <= 0.0 {
        r2 = 0.99999999999 * rl2;
    }
    let ffac1: f64 = r1 / rl1;
    let ffac2: f64 = r2 / rl2;

    let rdisc1: f64 = if model.rdisc1.value > 0.0 {
        model.rdisc1.value
    } else {
        r1
    };
    let rdisc2: f64 = if model.rdisc2.value > 0.0 {
        model.rdisc2.value
    } else {
        model.radius_spot.value
    };

    // Direction chosen so the stream is visible at all inclinations < 90.
    let dirn = Vec3::new(0.0, 0.0, 1.0);

    let add_eclipses = |posn: &Vec3| -> Result<Etype, RocheError> {
        let mut eclipses = Etype::new();
        star_eclipse(
            &roche_context1,
            r1,
            ffac1,
            iangle,
            posn,
            model.delta_phase,
            model.roche1,
            Star::Primary,
            &mut eclipses,
        )?;
        star_eclipse(
            &roche_context2,
            r2,
            ffac2,
            iangle,
            posn,
            model.delta_phase,
            model.roche2,
            Star::Secondary,
            &mut eclipses,
        )?;
        if model.add_disc {
            for eclipse_pair in roche::disc_eclipse(
                iangle,
                rdisc1,
                rdisc2,
                model.beta_disc.value,
                model.height_disc.value,
                posn,
            )? {
                eclipses.push(eclipse_pair);
            }
        }
        Ok(eclipses)
    };

    let (mut posn, mut v) = roche::strinit(q)?;

    let mut stream: Vec<Point> = Vec::new();
    stream.push(Point::new(posn, dirn, 0.0, 1.0, add_eclipses(&posn)?));

    let rl1_point: f64 = roche::x_l1(q)?;
    let radius_spot: f64 = model.radius_spot.value;
    let nstream: usize = (((rl1_point - radius_spot) / 0.001) as usize).max(1);
    for i in 0..nstream {
        let radius: f64 = rl1_point + (radius_spot - rl1_point) * (i as f64 + 1.0) / nstream as f64;
        roche::stradv(q, &mut posn, &mut v, radius, 1.0e-10, 1.0e-3)?;
        stream.push(Point::new(posn, dirn, 0.0, 1.0, add_eclipses(&posn)?));
    }
    Ok(stream)
}

/// Fill a circle of the given radius (in pixels) into the indexed buffer,
/// clipped to the image bounds.
fn fill_circle(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    cx: f64,
    cy: f64,
    radius: f64,
    colour: u8,
) {
    let r = radius.max(0.5);
    let x_min = ((cx - r).floor().max(0.0)) as usize;
    let x_max = ((cx + r).ceil().min(width as f64 - 1.0)) as usize;
    let y_min = ((cy - r).floor().max(0.0)) as usize;
    let y_max = ((cy + r).ceil().min(height as f64 - 1.0)) as usize;
    if cx + r < 0.0 || cy + r < 0.0 || cx - r > width as f64 - 1.0 || cy - r > height as f64 - 1.0 {
        return;
    }
    let r2 = r * r;
    for y in y_min..=y_max {
        let dy = y as f64 - cy;
        let row = y * width;
        for x in x_min..=x_max {
            let dx = x as f64 - cx;
            if dx * dx + dy * dy <= r2 {
                buffer[row + x] = colour;
            }
        }
    }
}

struct FrameGeometry {
    earth: Vec3,
    xsky: Vec3,
    ysky: Vec3,
    phase_normed: f64,
    x_scale: f64,
    y_scale: f64,
}

impl FrameGeometry {
    fn new(scene: &Scene, options: &RenderOptions, phase: f64) -> Self {
        let earth = roche::set_earth_iangle(scene.iangle, phase);
        let (sinp, cosp) = (std::f64::consts::TAU * phase).sin_cos();
        let xsky = Vec3::new(sinp, cosp, 0.0);
        let ysky = earth.cross(&xsky);
        Self {
            earth,
            xsky,
            ysky,
            phase_normed: phase - phase.floor(),
            x_scale: (options.width - 1) as f64 / (options.x2 - options.x1),
            y_scale: (options.height - 1) as f64 / (options.y2 - options.y1),
        }
    }

    fn project(&self, options: &RenderOptions, scene: &Scene, position: &Vec3) -> (f64, f64) {
        let r: Vec3 = *position - scene.cofm;
        let sx: f64 = r.dot(&self.xsky);
        let sy: f64 = r.dot(&self.ysky);
        let px = (sx - options.x1) * self.x_scale;
        let py = (options.height - 1) as f64 - (sy - options.y1) * self.y_scale;
        (px, py)
    }
}

/// Draw every point of one component that faces the observer and is not
/// eclipsed at this phase. Points with a zero direction vector (the disc
/// edges in "visual" mode) are treated as always facing the observer.
fn draw_component(
    buffer: &mut [u8],
    scene: &Scene,
    options: &RenderOptions,
    geometry: &FrameGeometry,
    points: &[Point],
    colour: u8,
    radius: f64,
) {
    for point in points {
        let d = point.direction;
        let faces_observer =
            (d.x == 0.0 && d.y == 0.0 && d.z == 0.0) || geometry.earth.dot(&d) > 0.0;
        if faces_observer && point.is_visible_phase_normed(geometry.phase_normed) {
            let (px, py) = geometry.project(options, scene, &point.position);
            fill_circle(
                buffer,
                options.width,
                options.height,
                px,
                py,
                radius,
                colour,
            );
        }
    }
}

/// Render one frame into a palette-indexed buffer of `width * height` bytes.
pub fn render_frame_indexed(scene: &Scene, options: &RenderOptions, phase: f64) -> Vec<u8> {
    let width = options.width;
    let height = options.height;
    let mut buffer = vec![BACKGROUND; width * height];
    let geometry = FrameGeometry::new(scene, options, phase);

    let dot_radius: f64 = (width.min(height) as f64 / 400.0).max(1.0);

    // Depth-sort the two stars so the nearer one is drawn on top.
    let star1_in_front = geometry.earth.x < 0.0;
    let (back_star, back_colour, front_star, front_colour) = if star1_in_front {
        (scene.star2, STAR2, scene.star1, STAR1)
    } else {
        (scene.star1, STAR1, scene.star2, STAR2)
    };

    draw_component(
        &mut buffer,
        scene,
        options,
        &geometry,
        back_star,
        back_colour,
        dot_radius,
    );
    if scene.add_disc {
        draw_component(
            &mut buffer,
            scene,
            options,
            &geometry,
            scene.disc,
            DISC,
            dot_radius,
        );
        draw_component(
            &mut buffer,
            scene,
            options,
            &geometry,
            &scene.outer_edge,
            EDGES,
            dot_radius,
        );
        draw_component(
            &mut buffer,
            scene,
            options,
            &geometry,
            &scene.inner_edge,
            EDGES,
            dot_radius,
        );
    }
    if scene.add_spot {
        draw_component(
            &mut buffer,
            scene,
            options,
            &geometry,
            &scene.stream,
            STAR2,
            dot_radius,
        );
    }
    draw_component(
        &mut buffer,
        scene,
        options,
        &geometry,
        front_star,
        front_colour,
        dot_radius,
    );

    // Emphasize the bright spot at the end of the stream with a larger
    // marker whose size grows as it turns towards the observer, as in the
    // original program.
    if let Some(spot) = scene.stream.last().filter(|_| scene.add_spot) {
        let cosbs: f64 = geometry.earth.dot(&spot.position);
        if cosbs > 0.0 && spot.is_visible_phase_normed(geometry.phase_normed) {
            let (px, py) = geometry.project(options, scene, &spot.position);
            let radius = (0.5 + 3.5 * cosbs) * width.min(height) as f64 / 200.0;
            fill_circle(&mut buffer, width, height, px, py, radius, STAR2);
        }
    }

    buffer
}

/// Sky-plane coordinates of every visible point of one component at the
/// given phase geometry, as (x, y) pairs relative to the centre of mass.
fn project_component(scene: &Scene, geometry: &FrameGeometry, points: &[Point]) -> Vec<[f64; 2]> {
    let mut projected = Vec::new();
    for point in points {
        let d = point.direction;
        let faces_observer =
            (d.x == 0.0 && d.y == 0.0 && d.z == 0.0) || geometry.earth.dot(&d) > 0.0;
        if faces_observer && point.is_visible_phase_normed(geometry.phase_normed) {
            let r: Vec3 = point.position - scene.cofm;
            projected.push([r.dot(&geometry.xsky), r.dot(&geometry.ysky)]);
        }
    }
    projected
}

/// The visible points of every component at one phase, projected onto the
/// sky plane. Components are returned in draw order; absent components
/// (no disc / no spot) come back as empty vectors. The bright spot is the
/// last stream point paired with its `cosbs` size factor when visible.
pub type ScenePoints = (
    Vec<[f64; 2]>,
    Vec<[f64; 2]>,
    Vec<[f64; 2]>,
    Vec<[f64; 2]>,
    Vec<[f64; 2]>,
    Vec<[f64; 2]>,
    Option<([f64; 2], f64)>,
);

pub fn scene_points(scene: &Scene, phase: f64) -> ScenePoints {
    // Any options with valid limits produce the same geometry; the sky
    // projection does not depend on the raster settings.
    let options = RenderOptions {
        width: 2,
        height: 2,
        x1: -1.0,
        x2: 1.0,
        y1: -1.0,
        y2: 1.0,
        reverse: true,
        sdob: false,
    };
    let geometry = FrameGeometry::new(scene, &options, phase);

    let spot = scene
        .stream
        .last()
        .filter(|_| scene.add_spot)
        .and_then(|spot| {
            let cosbs: f64 = geometry.earth.dot(&spot.position);
            if cosbs > 0.0 && spot.is_visible_phase_normed(geometry.phase_normed) {
                let r: Vec3 = spot.position - scene.cofm;
                Some(([r.dot(&geometry.xsky), r.dot(&geometry.ysky)], cosbs))
            } else {
                None
            }
        });

    (
        project_component(scene, &geometry, scene.star1),
        project_component(scene, &geometry, scene.star2),
        project_component(scene, &geometry, scene.disc),
        project_component(scene, &geometry, &scene.outer_edge),
        project_component(scene, &geometry, &scene.inner_edge),
        project_component(scene, &geometry, &scene.stream),
        spot,
    )
}

/// Expand a palette-indexed buffer into packed RGB bytes.
pub fn indexed_to_rgb(buffer: &[u8], palette: &[[u8; 3]; NCOLOURS]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(buffer.len() * 3);
    for &index in buffer {
        rgb.extend_from_slice(&palette[index as usize]);
    }
    rgb
}

/// The uniformly spaced phases of an animation running from `phase1` to
/// `phase2` inclusive; a single frame sits at `phase1`.
pub fn phase_grid(phase1: f64, phase2: f64, nphase: usize) -> Vec<f64> {
    (0..nphase)
        .map(|i| {
            if nphase == 1 {
                phase1
            } else {
                phase1 + (phase2 - phase1) * i as f64 / (nphase - 1) as f64
            }
        })
        .collect()
}

fn io_error(err: impl std::fmt::Display, context: &str) -> RocheError {
    RocheError::ParameterError(format!("{context}: {err}"))
}

/// Write one palette-indexed frame as an indexed-colour PNG.
pub fn write_png(path: &Path, buffer: &[u8], options: &RenderOptions) -> Result<(), RocheError> {
    let file = std::fs::File::create(path).map_err(|e| io_error(e, "cannot create PNG"))?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, options.width as u32, options.height as u32);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_palette(options.palette().concat());
    let mut png_writer = encoder
        .write_header()
        .map_err(|e| io_error(e, "cannot write PNG header"))?;
    png_writer
        .write_image_data(buffer)
        .map_err(|e| io_error(e, "cannot write PNG data"))?;
    Ok(())
}

/// Render an animated GIF. Frames are rendered in parallel in chunks so the
/// peak memory use stays bounded while the (sequential) encoder keeps up.
pub fn write_gif(
    path: &Path,
    scene: &Scene,
    options: &RenderOptions,
    phases: &[f64],
    fps: usize,
) -> Result<(), RocheError> {
    let file = std::fs::File::create(path).map_err(|e| io_error(e, "cannot create GIF"))?;
    let writer = std::io::BufWriter::new(file);
    let palette = options.palette().concat();
    let mut encoder = gif::Encoder::new(
        writer,
        options.width as u16,
        options.height as u16,
        &palette,
    )
    .map_err(|e| io_error(e, "cannot write GIF header"))?;
    encoder
        .set_repeat(gif::Repeat::Infinite)
        .map_err(|e| io_error(e, "cannot set GIF repeat"))?;

    let delay_cs = (100 / fps.max(1)).max(1) as u16;
    let chunk_size = rayon::current_num_threads().max(1) * 4;
    for chunk in phases.chunks(chunk_size) {
        let frames: Vec<Vec<u8>> = chunk
            .par_iter()
            .map(|&phase| render_frame_indexed(scene, options, phase))
            .collect();
        for buffer in frames {
            let mut frame = gif::Frame {
                width: options.width as u16,
                height: options.height as u16,
                delay: delay_cs,
                buffer: std::borrow::Cow::Owned(buffer),
                ..gif::Frame::default()
            };
            frame.make_lzw_pre_encoded();
            encoder
                .write_lzw_pre_encoded_frame(&frame)
                .map_err(|e| io_error(e, "cannot write GIF frame"))?;
        }
    }
    Ok(())
}

static MP4_RUN_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Render an MP4 by writing PNG frames to a temporary directory (in
/// parallel) and invoking the system `ffmpeg` binary, exactly like the
/// original C++ tool. Fails with a clear message if `ffmpeg` is missing.
pub fn write_mp4(
    path: &Path,
    scene: &Scene,
    options: &RenderOptions,
    phases: &[f64],
    fps: usize,
) -> Result<(), RocheError> {
    let run = MP4_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmpdir: PathBuf =
        std::env::temp_dir().join(format!("lroche_frames_{}_{}", std::process::id(), run));
    std::fs::create_dir_all(&tmpdir).map_err(|e| io_error(e, "cannot create frame dir"))?;

    let result = (|| -> Result<(), RocheError> {
        phases
            .par_iter()
            .enumerate()
            .try_for_each(|(i, &phase)| -> Result<(), RocheError> {
                let buffer = render_frame_indexed(scene, options, phase);
                write_png(&tmpdir.join(format!("frame_{i:05}.png")), &buffer, options)
            })?;

        let pattern = tmpdir.join("frame_%05d.png");
        let status = Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-v", "error", "-framerate"])
            .arg(fps.to_string())
            .arg("-i")
            .arg(&pattern)
            .args([
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ])
            .arg(path)
            .status();

        match status {
            Ok(status) if status.success() => Ok(()),
            Ok(status) => Err(RocheError::ParameterError(format!(
                "ffmpeg failed with {status}."
            ))),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Err(RocheError::ParameterError(
                    "ffmpeg not found on PATH; install it or render a .gif instead.".to_string(),
                ))
            }
            Err(err) => Err(io_error(err, "cannot run ffmpeg")),
        }
    })();

    let _ = std::fs::remove_dir_all(&tmpdir);
    result
}
