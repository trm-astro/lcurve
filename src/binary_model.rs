use crate::comp_gravity::{comp_gravity1, comp_gravity2};
use crate::comp_light::{comp_bright_spot, comp_disc, comp_disc_edge, comp_star1, comp_star2};
use crate::comp_radius::comp_radius;
use crate::ginterp::Ginterp;
use crate::grid::Grid;
use crate::ldc::LDC;
use crate::model::{Entry, Model, ModelUpdate};
use crate::set_bright_spot_grid::set_bright_spot_grid;
use crate::set_disc_continuum::{set_disc_continuum, set_edge_continuum};
use crate::set_disc_grid::{set_disc_edge_grid, set_disc_grid};
use crate::set_star_continuum::set_star_continuum;
use crate::set_star_grid::set_star_grid;
use crate::visualize::{self, RenderOptions};
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use rayon::prelude::*;
use roche::constants::{C, DAY};
use roche::errors::RocheError;
use roche::{self, Etype, Star, disc_eclipse};
use serde_pyobject::from_pyobject;
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::path::PathBuf;

#[pyclass]
pub struct LightCurve {
    #[pyo3(get)]
    pub star1: Py<PyArray1<f64>>,

    #[pyo3(get)]
    pub star2: Py<PyArray1<f64>>,

    #[pyo3(get)]
    pub disc: Py<PyArray1<f64>>,

    #[pyo3(get)]
    pub disc_edge: Py<PyArray1<f64>>,

    #[pyo3(get)]
    pub bright_spot: Py<PyArray1<f64>>,

    #[pyo3(get)]
    pub total: Py<PyArray1<f64>>,

    #[pyo3(get)]
    pub scale_factor: f64,

    #[pyo3(get)]
    pub star1_contribution: f64,

    #[pyo3(get)]
    pub logg1: Option<f64>,

    #[pyo3(get)]
    pub logg2: Option<f64>,

    #[pyo3(get)]
    pub rva1: f64,

    #[pyo3(get)]
    pub rva2: f64,

    #[pyo3(get)]
    pub chi2: Option<f64>,

    #[pyo3(get)]
    pub log_prob: Option<f64>,
}

///
/// :class:`BinaryModel` is a class to contain the model and grids for a close
/// binary system.
///
/// An instance of :class:`BinaryModel` can be initialised either from an
/// `lcurve` .mod file using :meth:`BinaryModel.from_file` or from an instance
/// of :class:`lcurve.Model` using :meth:`BinaryModel.from_model`.
///
/// Parameters can be updated by supplying a python dictionary of parameter
/// key: value pairs to :meth:`BinaryModel.update`.
///
#[pyclass]
pub struct BinaryModel {
    #[pyo3(get)]
    star1_coarse_grid: Grid,
    #[pyo3(get)]
    star2_coarse_grid: Grid,
    #[pyo3(get)]
    star1_fine_grid: Grid,
    #[pyo3(get)]
    star2_fine_grid: Grid,
    #[pyo3(get)]
    disc_grid: Grid,
    #[pyo3(get)]
    disc_edge_grid: Grid,
    #[pyo3(get)]
    bright_spot_grid: Grid,
    gint: Ginterp,
    rlens1: f64,
    model_beaming1: bool,
    model_beaming2: bool,

    #[pyo3(get)]
    pub model: Model,
}

#[pymethods]
impl BinaryModel {

    /// Create a :class:`BinaryModel` from an lcurve .mod file.
    ///
    /// Parameters:
    ///   filename (str): the path to the .mod file
    ///
    /// Returns:
    ///   BinaryModel: class instance
    /// 
    #[staticmethod]
    pub fn from_file(filename: &str) -> PyResult<Self> {
        let model = Model::from_file(filename).map_err(pyo3::exceptions::PyIOError::new_err)?;
        let (
            star1_coarse_grid,
            star2_coarse_grid,
            star1_fine_grid,
            star2_fine_grid,
            disc_grid,
            disc_edge_grid,
            bright_spot_grid,
            gint,
            rlens1,
            model_beaming1,
            model_beaming2,
        ) = build_grids(&model)?;

        Ok(Self {
            star1_coarse_grid,
            star2_coarse_grid,
            star1_fine_grid,
            star2_fine_grid,
            disc_grid,
            disc_edge_grid,
            bright_spot_grid,
            gint,
            rlens1,
            model_beaming1,
            model_beaming2,
            model,
        })
    }

    #[staticmethod]
    pub fn from_model(model: Model) -> PyResult<Self> {
        let (
            star1_coarse_grid,
            star2_coarse_grid,
            star1_fine_grid,
            star2_fine_grid,
            disc_grid,
            disc_edge_grid,
            bright_spot_grid,
            gint,
            rlens1,
            model_beaming1,
            model_beaming2,
        ) = build_grids(&model)?;

        Ok(Self {
            star1_coarse_grid,
            star2_coarse_grid,
            star1_fine_grid,
            star2_fine_grid,
            disc_grid,
            disc_edge_grid,
            bright_spot_grid,
            gint,
            rlens1,
            model_beaming1,
            model_beaming2,
            model,
        })
    }

    ///
    /// Method to update :class:`BinaryModel.Model` from a supplied Python dictionary and
    /// rebuild grids if the geometry has changed, otherwise just updating the
    /// continuum of the current grid.
    /// 
    pub fn update(&mut self, _py: Python, dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let upd: ModelUpdate = from_pyobject(dict.clone())?;
        let grid_changed = upd.grid_changed();
        self.model.apply_update(upd)?;
        if grid_changed {
            (
                self.star1_coarse_grid,
                self.star2_coarse_grid,
                self.star1_fine_grid,
                self.star2_fine_grid,
                self.disc_grid,
                self.disc_edge_grid,
                self.bright_spot_grid,
                self.gint,
                self.rlens1,
                self.model_beaming1,
                self.model_beaming2
            ) = build_grids(&self.model)?;
        } else {
            self.reset_grid_continuum()?;
        }
        Ok(())
    }

    ///
    /// grid options are:
    /// "star1_fine",
    /// "star1_coarse",
    /// "star2_fine",
    /// "star2_coarse",
    /// "disc",
    /// "disc_edge",
    /// "bright_spot"
    /// 
    // pub fn set_grid_fluxes(&mut self, grid: &str, fluxes: Vec<f32>) -> Result<(), RocheError> {
    //     let chosen_grid = match grid {
    //         "star1_fine" => &mut self.star1_fine_grid,
    //         "star1_coarse" => &mut self.star1_coarse_grid,
    //         "star2_fine" => &mut self.star2_fine_grid,
    //         "star2_coarse" => &mut self.star2_coarse_grid,
    //         "disc" => &mut self.disc_grid,
    //         "disc_edge" => &mut self.disc_edge_grid,
    //         "bright_spot" => &mut self.bright_spot_grid,
    //         _ => return Err(RocheError::ParameterError("Not a valid grid.".to_string())),
    //     };

    //     apply_fluxes(chosen_grid, fluxes)?;
    //     self.gint = set_ginterp(
    //         &self.model,
    //         self.rlens1,
    //         &self.star1_coarse_grid,
    //         &self.star2_coarse_grid,
    //         &self.star1_fine_grid,
    //         &self.star2_fine_grid,
    //     )?;

    //     Ok(())
    // }

    ///
    /// Computes a model light curve for an array of times and exposure times
    /// for the current parameters defined in :class:`BinaryModel.Model`.
    /// 
    /// Parameters:
    /// 
    /// * `time`: Array of times
    /// * `t_exp`: Array of exposure times (same units as `time`)
    /// * `n_div`: (Optional) number of exposure subdivisions to use to model\
    ///             smearing from finite exposure times. 
    /// * `flux`: (Optional) flux of data to enable automatic scaling of the model
    ///             as well as for calculations of chi2 and log_prob.
    /// * `flux_err`: (Optional) flux uncertainty of data to enable automatic
    ///             scaling of the model as well as for calculations of chi2 and
    ///             log_prob.
    /// * `weight`: (Optional) weights for autoscaling, chi2, and log_prob
    /// * `scale_factor`: (Optional) Scale factor to multiply light curve model
    ///             by. Prevents autoscaling
    /// 
    /// Returns:
    ///     
    ///     :class:`LightCurve`
    /// 
    #[pyo3(signature = (
        time,
        t_exp,
        n_div=None,
        flux=None,
        flux_err=None,
        weight=None,
        scale_factor=None,
    ))]
    pub fn compute_light_curve(
        &self,
        py: Python,
        time: PyReadonlyArray1<f64>,
        t_exp: PyReadonlyArray1<f64>,
        n_div: Option<PyReadonlyArray1<f64>>,
        flux: Option<PyReadonlyArray1<f64>>,
        flux_err: Option<PyReadonlyArray1<f64>>,
        weight: Option<PyReadonlyArray1<f64>>,
        scale_factor: Option<f64>,
    ) -> PyResult<LightCurve> {

        let time: &[f64] = time.as_slice()?;
        let t_exp: &[f64] = t_exp.as_slice()?;
        // let n_div: &[f64] = n_div.as_slice()?;
        let n: usize = time.len();

        let n_div_default;
        let n_div = if let Some(ref ndiv) = n_div {
            ndiv.as_slice()?
        } else {
            n_div_default = vec![1.0_f64; n];
            &n_div_default
        };
        let flux = match &flux {
            Some(f) => Some(f.as_slice()?),
            None => None,
        };
        let flux_err = match &flux_err {
            Some(f) => Some(f.as_slice()?),
            None => None,
        };
        let weight = match &weight {
            Some(f) => Some(f.as_slice()?),
            None => None,
        };

        let mut star1 = vec![0.0; n];
        let mut star2 = vec![0.0; n];
        let mut disc = vec![0.0; n];
        let mut disc_edge = vec![0.0; n];
        let mut bright_spot = vec![0.0; n];
        let mut total = vec![0.0; n];

        let ldc1: LDC = self.model.get_ldc1();
        let ldc2: LDC = self.model.get_ldc2();

        let mut xmin: f64 = time[0];
        let mut xmax: f64 = time[0];
        for t in time {
            xmin = if *t > xmin {xmin} else {*t};
            xmax = if *t < xmax {xmax} else {*t};
        }

        let middle: f64 = (xmin + xmax) / 2.0;
        let range: f64 = (xmax - xmin) / 2.0;

        star1.par_iter_mut().enumerate().for_each(|(i, out)| {
            let phase: f64 = calc_phase(time[i], self.model.t0.value, self.model.period.value, self.model.pdot.value, self.model.deltat.value);
            let expose: f64 = t_exp[i] / self.model.period.value;
            let frac: f64 = (time[i] - middle) / range;
            let slfac: f64 = 1.0 + frac * (self.model.slope.value + frac * (self.model.quad.value + frac * self.model.cube.value));
            *out = slfac * self.compute_star1_flux(phase, &ldc1, expose, n_div[i] as i32);
        });

        star2.par_iter_mut().enumerate().for_each(|(i, out)| {
            let phase: f64 = calc_phase(time[i], self.model.t0.value, self.model.period.value, self.model.pdot.value, self.model.deltat.value);
            let expose: f64 = t_exp[i] / self.model.period.value;
            let frac: f64 = (time[i] - middle) / range;
            let slfac: f64 = 1.0 + frac * (self.model.slope.value + frac * (self.model.quad.value + frac * self.model.cube.value));
            *out = slfac * self.compute_star2_flux(phase, &ldc2, expose, n_div[i] as i32);
        });

        disc.par_iter_mut().enumerate().for_each(|(i, out)| {
            let phase: f64 = calc_phase(time[i], self.model.t0.value, self.model.period.value, self.model.pdot.value, self.model.deltat.value);
            let expose: f64 = t_exp[i] / self.model.period.value;
            let frac: f64 = (time[i] - middle) / range;
            let slfac: f64 = 1.0 + frac * (self.model.slope.value + frac * (self.model.quad.value + frac * self.model.cube.value));
            *out = slfac * self.compute_disc_flux(phase, expose, n_div[i] as i32);
        });

        disc_edge.par_iter_mut().enumerate().for_each(|(i, out)| {
            let phase: f64 = calc_phase(time[i], self.model.t0.value, self.model.period.value, self.model.pdot.value, self.model.deltat.value);
            let expose: f64 = t_exp[i] / self.model.period.value;
            let frac: f64 = (time[i] - middle) / range;
            let slfac: f64 = 1.0 + frac * (self.model.slope.value + frac * (self.model.quad.value + frac * self.model.cube.value));
            *out = slfac * self.compute_disc_edge_flux(phase, expose, n_div[i] as i32);
        });

        bright_spot.par_iter_mut().enumerate().for_each(|(i, out)| {
            let phase: f64 = calc_phase(time[i], self.model.t0.value, self.model.period.value, self.model.pdot.value, self.model.deltat.value);
            let expose: f64 = t_exp[i] / self.model.period.value;
            let frac: f64 = (time[i] - middle) / range;
            let slfac: f64 = 1.0 + frac * (self.model.slope.value + frac * (self.model.quad.value + frac * self.model.cube.value));
            *out = slfac * self.compute_bright_spot_flux(phase, expose, n_div[i] as i32);
        });

        let mut star1_contribution: f64 = self.compute_star1_flux(0.5, &ldc1, 0.0, 1);

        for i in 0..time.len() {
            total[i] = star1[i] + star2[i] + disc[i] + disc_edge[i] + bright_spot[i];
        }

        let scale_factor = match (scale_factor, flux, flux_err) {
            (Some(scale_factor), _, _) => scale_factor,
            (None, Some(flux), Some(flux_err)) => rescale(flux, flux_err, weight, &total),
            _ => 1.0,
        };

        for i in 0..time.len() {
            star1[i] *= scale_factor;
            star2[i] *= scale_factor;
            disc[i] *= scale_factor;
            disc_edge[i] *= scale_factor;
            bright_spot[i] *= scale_factor;
            total[i] *= scale_factor;
        }
        star1_contribution *= scale_factor;

        let (chisq, log_prob) = match (flux, flux_err) {
            (Some(flux), Some(flux_err)) => {
                let (chisq, log_prob) = chisq_log_prob(flux, flux_err, weight, &total);
                (Some(chisq), Some(log_prob))
            }
            _ => (None, None),
        };

        let (logg1, logg2) = if self.model.velocity_scale.defined {
            let logg1 = comp_gravity1(&self.model, &self.star1_fine_grid)?;
            let logg2 = comp_gravity2(&self.model, &self.star2_fine_grid)?;
            (Some(logg1), Some(logg2))
        } else {
            (None, None)
        };

        let rva1: f64 = if self.model.roche1 {
            comp_radius(&self.star1_coarse_grid.points, Star::Primary)
        } else {
            self.model.r1.value
        };
        let rva2: f64 = comp_radius(&self.star2_coarse_grid.points, Star::Secondary);

        Ok(LightCurve {
            star1: star1.into_pyarray(py).unbind(),
            star2: star2.into_pyarray(py).unbind(),
            disc: disc.into_pyarray(py).unbind(),
            disc_edge: disc_edge.into_pyarray(py).unbind(),
            bright_spot: bright_spot.into_pyarray(py).unbind(),
            total: total.into_pyarray(py).unbind(),
            scale_factor,
            star1_contribution,
            logg1,
            logg2,
            rva1,
            rva2,
            chi2: chisq,
            log_prob,
        })
    }

    ///
    /// Render the system at a single orbital phase, projected onto the
    /// plane of the sky, in the style of the original LCURVE `visualise`
    /// program. Returns the image as a (height, width, 3) uint8 numpy
    /// array; if `path` is given the image is also written there as a PNG.
    ///
    /// Keyword arguments:
    ///     width: image width in pixels (default 800).
    ///     height: image height in pixels; derived from the aspect ratio
    ///         of the plot limits when omitted.
    ///     x1, x2, y1, y2: sky-plane plot limits in units of the binary
    ///         separation (defaults -2, 2, -2, 2).
    ///     reverse: white background with dark colours if True (default),
    ///         black background with bright colours if False.
    ///     sdob: swap the colours of the two stars (default False).
    ///     path: optional filename for a PNG copy of the frame.
    ///
    #[pyo3(signature = (
        phase,
        *,
        width=800,
        height=None,
        x1=-2.0,
        x2=2.0,
        y1=-2.0,
        y2=2.0,
        reverse=true,
        sdob=false,
        path=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &self,
        py: Python,
        phase: f64,
        width: usize,
        height: Option<usize>,
        x1: f64,
        x2: f64,
        y1: f64,
        y2: f64,
        reverse: bool,
        sdob: bool,
        path: Option<PathBuf>,
    ) -> PyResult<Py<PyArray3<u8>>> {
        let options = RenderOptions::new(width, height, x1, x2, y1, y2, reverse, sdob, false)?;
        let rgb = py.detach(|| -> Result<Vec<u8>, RocheError> {
            let scene = visualize::build_scene(
                &self.model,
                &self.star1_fine_grid,
                &self.star2_fine_grid,
                &self.disc_grid,
            )?;
            let buffer = visualize::render_frame_indexed(&scene, &options, phase);
            if let Some(ref path) = path {
                visualize::write_png(path, &buffer, &options)?;
            }
            Ok(visualize::indexed_to_rgb(&buffer, &options.palette()))
        })?;
        let array = Array3::from_shape_vec((options.height, options.width, 3), rgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py).unbind())
    }

    ///
    /// Project the visible points of every model component onto the plane
    /// of the sky at the given orbital phase, in units of the binary
    /// separation relative to the centre of mass. Returns a dict mapping
    /// "star1", "star2", "disc", "disc_edge_outer", "disc_edge_inner" and
    /// "stream" to (n, 2) float64 numpy arrays of (x, y) sky coordinates
    /// (empty for absent components), plus "bright_spot", which is either
    /// None or a ((x, y), cosbs) tuple where cosbs sets the marker size in
    /// the rendered frames. Useful for custom plotting, e.g. matplotlib.
    ///
    pub fn scene_points(&self, py: Python, phase: f64) -> PyResult<Py<PyDict>> {
        let scene = visualize::build_scene(
            &self.model,
            &self.star1_fine_grid,
            &self.star2_fine_grid,
            &self.disc_grid,
        )?;
        let (star1, star2, disc, outer_edge, inner_edge, stream, spot) =
            visualize::scene_points(&scene, phase);

        let to_array = |points: Vec<[f64; 2]>| -> PyResult<Py<PyArray2<f64>>> {
            let n = points.len();
            let flat: Vec<f64> = points.into_iter().flatten().collect();
            let array = Array2::from_shape_vec((n, 2), flat)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(array.into_pyarray(py).unbind())
        };

        let dict = PyDict::new(py);
        dict.set_item("star1", to_array(star1)?)?;
        dict.set_item("star2", to_array(star2)?)?;
        dict.set_item("disc", to_array(disc)?)?;
        dict.set_item("disc_edge_outer", to_array(outer_edge)?)?;
        dict.set_item("disc_edge_inner", to_array(inner_edge)?)?;
        dict.set_item("stream", to_array(stream)?)?;
        dict.set_item("bright_spot", spot.map(|(xy, cosbs)| ((xy[0], xy[1]), cosbs)))?;
        Ok(dict.unbind())
    }

    ///
    /// Render an animation of the orbit between two phases and write it to
    /// `path`. The format is chosen from the file extension: `.gif` is
    /// encoded natively, `.mp4` requires the `ffmpeg` binary on the PATH.
    /// Frames are rendered in parallel.
    ///
    /// Keyword arguments:
    ///     phase1, phase2: first and last orbital phase (defaults 0 and 1).
    ///     nphase: number of frames (default 300).
    ///     fps: frames per second, 1 to 240 (default 25).
    ///     width, height, x1, x2, y1, y2, reverse, sdob: as render_frame.
    ///
    #[pyo3(signature = (
        path,
        *,
        phase1=0.0,
        phase2=1.0,
        nphase=300,
        fps=25,
        width=800,
        height=None,
        x1=-2.0,
        x2=2.0,
        y1=-2.0,
        y2=2.0,
        reverse=true,
        sdob=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn render_animation(
        &self,
        py: Python,
        path: PathBuf,
        phase1: f64,
        phase2: f64,
        nphase: usize,
        fps: usize,
        width: usize,
        height: Option<usize>,
        x1: f64,
        x2: f64,
        y1: f64,
        y2: f64,
        reverse: bool,
        sdob: bool,
    ) -> PyResult<()> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let is_mp4 = match extension.as_deref() {
            Some("gif") => false,
            Some("mp4") => true,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "path must end in .gif or .mp4",
                ));
            }
        };
        if nphase < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nphase must be at least 1",
            ));
        }
        if !(1..=240).contains(&fps) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fps must be between 1 and 240",
            ));
        }
        let options = RenderOptions::new(width, height, x1, x2, y1, y2, reverse, sdob, is_mp4)?;
        py.detach(|| -> Result<(), RocheError> {
            let scene = visualize::build_scene(
                &self.model,
                &self.star1_fine_grid,
                &self.star2_fine_grid,
                &self.disc_grid,
            )?;
            let phases = visualize::phase_grid(phase1, phase2, nphase);
            if is_mp4 {
                visualize::write_mp4(&path, &scene, &options, &phases, fps)
            } else {
                visualize::write_gif(&path, &scene, &options, &phases, fps)
            }
        })?;
        Ok(())
    }
}

impl BinaryModel {
    fn compute_star1_flux(&self, phase: f64, ldc1: &LDC, expose: f64, n_div: i32) -> f64 {
        comp_star1(
            self.model.iangle.value,
            ldc1,
            phase,
            expose,
            n_div,
            self.model.q.value,
            self.model.beam_factor1.value,
            self.model.velocity_scale.value,
            self.model_beaming1,
            &self.gint,
            &self.star1_fine_grid,
            &self.star1_coarse_grid,
        )
    }

    fn compute_star2_flux(&self, phase: f64, ldc2: &LDC, expose: f64, n_div: i32) -> f64 {
        comp_star2(
            self.model.iangle.value,
            ldc2,
            phase,
            expose,
            n_div,
            self.model.q.value,
            self.model.beam_factor2.value,
            self.model.velocity_scale.value,
            self.model_beaming2,
            self.model.glens1,
            self.rlens1,
            &self.gint,
            &self.star2_fine_grid,
            &self.star2_coarse_grid,
        )
    }

    fn compute_disc_flux(&self, phase: f64, expose: f64, n_div: i32) -> f64 {
        comp_disc(
            self.model.iangle.value,
            self.model.lin_limb_disc.value,
            self.model.quad_limb_disc.value,
            phase,
            expose,
            n_div,
            &self.disc_grid,
        )
    }

    fn compute_disc_edge_flux(&self, phase: f64, expose: f64, n_div: i32) -> f64 {
        comp_disc_edge(
            self.model.iangle.value,
            self.model.lin_limb_disc.value,
            self.model.quad_limb_disc.value,
            phase,
            expose,
            n_div,
            &self.disc_edge_grid,
        )
    }

    fn compute_bright_spot_flux(&self, phase: f64, expose: f64, n_div: i32) -> f64 {
        comp_bright_spot(
            self.model.iangle.value,
            phase,
            expose,
            n_div,
            &self.bright_spot_grid,
        )
    }

    fn reset_grid_continuum(&mut self) -> Result<(), RocheError> {
        self.model.validate()?;
        let (_r1, mut r2) = self.model.get_r1r2();
        let rl2: f64 = 1.0 - roche::x_l1_2(self.model.q.value, self.model.spin2.value)?;
        if r2 < 0.0 {
            r2 = rl2;
        } else if r2 > rl2 {
            panic!("Secondary is larger than Roche Lobe.")
        }

        self.model_beaming1 = self.model.beam_factor1.defined && self.model.velocity_scale.defined;
        self.model_beaming2 = self.model.beam_factor2.defined && self.model.velocity_scale.defined;


        set_star_continuum(
            &self.model,
            &mut self.star1_fine_grid,
            &mut self.star2_fine_grid,
        )?;
        set_star_continuum(
            &self.model,
            &mut self.star1_coarse_grid,
            &mut self.star2_coarse_grid,
        )?;

        self.gint = set_ginterp(
            &self.model,
            self.rlens1,
            &self.star1_coarse_grid,
            &self.star2_coarse_grid,
            &self.star1_fine_grid,
            &self.star2_fine_grid,
        )?;

        if self.model.add_disc {
            let rdisc2 = if self.model.rdisc2.value > 0.0 {
                self.model.rdisc2.value
            } else {
                self.model.radius_spot.value
            };

            // Set the surface brightness of the disc
            set_disc_continuum(
                rdisc2,
                self.model.temp_disc.value,
                self.model.texp_disc.value,
                self.model.wavelength,
                &mut self.disc_grid,
            );

            // Set the surface brightness of outer edge, accounting for
            // irradiation by star 2
            set_edge_continuum(
                self.model.temp_edge.value,
                r2,
                self.model.t2.value.abs(),
                self.model.absorb_edge.value,
                self.model.wavelength,
                &mut self.disc_edge_grid,
            );
        }

        if self.model.add_spot {
            self.bright_spot_grid = set_bright_spot_grid(&self.model)?;
        }
        Ok(())
    }
}

fn build_grids(
    model: &Model,
) -> Result<
    (
        Grid,
        Grid,
        Grid,
        Grid,
        Grid,
        Grid,
        Grid,
        Ginterp,
        f64,
        bool,
        bool,
    ),
    RocheError,
> {
    model.validate()?;
    let mut star1_fine_grid = set_star_grid(model, Star::Primary, true)?;
    let mut star2_fine_grid = set_star_grid(model, Star::Secondary, true)?;
    let mut star1_coarse_grid: Grid;
    let mut star2_coarse_grid: Grid;

    let (r1, mut r2) = model.get_r1r2();
    let rl2: f64 = 1.0 - roche::x_l1_2(model.q.value, model.spin2.value)?;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        panic!("Secondary is larger than Roche Lobe.")
    }

    set_star_continuum(model, &mut star1_fine_grid, &mut star2_fine_grid)?;

    if model.nlat1f == model.nlat1c {
        star1_coarse_grid = star1_fine_grid.clone();
    } else {
        star1_coarse_grid = set_star_grid(model, Star::Primary, false)?;
    }

    let copy2: bool = (model.nlat2f == model.nlat2c)
        && (!model.npole || r1 >= r2 || (model.nlatfill == 0 && model.nlngfill == 0));

    if copy2 {
        star2_coarse_grid = star2_fine_grid.clone();
    } else {
        star2_coarse_grid = set_star_grid(model, Star::Secondary, false)?
    }

    if model.nlat1c != model.nlat1f || !copy2 {
        set_star_continuum(model, &mut star1_coarse_grid, &mut star2_coarse_grid)?;
    }

    let mut disc_grid: Grid = Grid::new(vec![]);
    let mut disc_edge_grid: Grid = Grid::new(vec![]);
    let mut bright_spot_grid: Grid = Grid::new(vec![]);

    let mut rlens1 = 0.0;
    if model.glens1 {
        let gm: f64 = (1000.0 * model.velocity_scale.value).powi(3) * model.tperiod * DAY / TAU;
        let a: f64 =
            (gm / ((TAU / DAY / model.tperiod) * (TAU / DAY / model.tperiod))).powf(1.0 / 3.0);
        rlens1 = 4.0 * gm / (1.0 + model.q.value) / a / (C * C);
    }

    let model_beaming1: bool = model.beam_factor1.defined && model.velocity_scale.defined;
    let model_beaming2: bool = model.beam_factor2.defined && model.velocity_scale.defined;

    let gint = set_ginterp(
        &model,
        rlens1,
        &star1_coarse_grid,
        &star2_coarse_grid,
        &star1_fine_grid,
        &star2_fine_grid,
    )?;

    if model.add_disc {
        disc_grid = set_disc_grid(model)?;
        disc_edge_grid = set_disc_edge_grid(model, true, false)?;

        let rdisc1 = if model.rdisc1.value > 0.0 {
            model.rdisc1.value
        } else {
            r1
        };
        let rdisc2 = if model.rdisc2.value > 0.0 {
            model.rdisc2.value
        } else {
            model.radius_spot.value
        };

        let mut eclipses: Etype;
        if model.opaque {
            for point in &mut star1_fine_grid.points {
                eclipses = disc_eclipse(
                    model.iangle.value,
                    rdisc1,
                    rdisc2,
                    model.beta_disc.value,
                    model.height_disc.value,
                    &point.position,
                )?;
                for eclipse_pair in eclipses {
                    point.eclipse.push(eclipse_pair);
                }
            }

            for point in &mut star1_coarse_grid.points {
                eclipses = disc_eclipse(
                    model.iangle.value,
                    rdisc1,
                    rdisc2,
                    model.beta_disc.value,
                    model.height_disc.value,
                    &point.position,
                )?;
                for eclipse_pair in eclipses {
                    point.eclipse.push(eclipse_pair);
                }
            }

            for point in &mut star2_fine_grid.points {
                eclipses = disc_eclipse(
                    model.iangle.value,
                    rdisc1,
                    rdisc2,
                    model.beta_disc.value,
                    model.height_disc.value,
                    &point.position,
                )?;
                for eclipse_pair in eclipses {
                    point.eclipse.push(eclipse_pair);
                }
            }

            for point in &mut star2_coarse_grid.points {
                eclipses = disc_eclipse(
                    model.iangle.value,
                    rdisc1,
                    rdisc2,
                    model.beta_disc.value,
                    model.height_disc.value,
                    &point.position,
                )?;
                for eclipse_pair in eclipses {
                    point.eclipse.push(eclipse_pair);
                }
            }
        }

        // Set the surface brightness of the disc
        set_disc_continuum(
            rdisc2,
            model.temp_disc.value,
            model.texp_disc.value,
            model.wavelength,
            &mut disc_grid,
        );

        // Set the surface brightness of outer edge, accounting for
        // irradiation by star 2
        set_edge_continuum(
            model.temp_edge.value,
            r2,
            model.t2.value.abs(),
            model.absorb_edge.value,
            model.wavelength,
            &mut disc_edge_grid,
        );
    }

    if model.add_spot {
        bright_spot_grid = set_bright_spot_grid(model)?;
    }

    Ok((
        star1_coarse_grid,
        star2_coarse_grid,
        star1_fine_grid,
        star2_fine_grid,
        disc_grid,
        disc_edge_grid,
        bright_spot_grid,
        gint,
        rlens1,
        model_beaming1,
        model_beaming2,
    ))
}

pub fn set_ginterp(
    model: &Model,
    rlens1: f64,
    star1c: &Grid,
    star2c: &Grid,
    star1f: &Grid,
    star2f: &Grid,
) -> Result<Ginterp, RocheError> {
    let (r1, mut r2) = model.get_r1r2();
    let rl2: f64 = 1.0 - roche::x_l1_2(model.q.value, model.spin2.value)?;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        panic!("Secondary is larger than Roche Lobe.")
    }

    let ldc1: LDC = model.get_ldc1();
    let ldc2: LDC = model.get_ldc2();

    let model_beaming1: bool = model.beam_factor1.defined && model.velocity_scale.defined;
    let model_beaming2: bool = model.beam_factor2.defined && model.velocity_scale.defined;

    let mut gint: Ginterp = Ginterp {
        phase1: model.phase1,
        phase2: model.phase2,
        scale11: 1.0,
        scale12: 1.0,
        scale21: 1.0,
        scale22: 1.0,
    };

    let copy2: bool = (model.nlat2f == model.nlat2c)
        && (!model.npole || r1 >= r2 || (model.nlatfill == 0 && model.nlngfill == 0));

    if model.nlat1c != model.nlat1f {
        let ff: f64 = comp_star1(
            model.iangle.value,
            &ldc1,
            0.9999999999 * model.phase1,
            0.0,
            1,
            model.q.value,
            model.beam_factor1.value,
            model.velocity_scale.value,
            model_beaming1,
            &gint,
            star1f,
            star1c,
        );
        let fc: f64 = comp_star1(
            model.iangle.value,
            &ldc1,
            1.0000000001 * model.phase1,
            0.0,
            1,
            model.q.value,
            model.beam_factor1.value,
            model.velocity_scale.value,
            model_beaming1,
            &gint,
            star1f,
            star1c,
        );
        gint.scale11 = ff / fc;

        let ff: f64 = comp_star1(
            model.iangle.value,
            &ldc1,
            1.0 - 0.9999999999 * model.phase1,
            0.0,
            1,
            model.q.value,
            model.beam_factor1.value,
            model.velocity_scale.value,
            model_beaming1,
            &gint,
            star1f,
            star1c,
        );
        let fc: f64 = comp_star1(
            model.iangle.value,
            &ldc1,
            1.0 - 1.0000000001 * model.phase1,
            0.0,
            1,
            model.q.value,
            model.beam_factor1.value,
            model.velocity_scale.value,
            model_beaming1,
            &gint,
            star1f,
            star1c,
        );
        gint.scale12 = ff / fc;
    }

    if !copy2 {
        let ff: f64 = comp_star2(
            model.iangle.value,
            &ldc2,
            1.0 - 1.0000000001 * model.phase2,
            0.0,
            1,
            model.q.value,
            model.beam_factor2.value,
            model.velocity_scale.value,
            model_beaming2,
            model.glens1,
            rlens1,
            &gint,
            star2f,
            star2c,
        );
        let fc: f64 = comp_star2(
            model.iangle.value,
            &ldc2,
            1.0 - 0.9999999999 * model.phase2,
            0.0,
            1,
            model.q.value,
            model.beam_factor2.value,
            model.velocity_scale.value,
            model_beaming2,
            model.glens1,
            rlens1,
            &gint,
            star2f,
            star2c,
        );
        gint.scale21 = ff / fc;

        let ff: f64 = comp_star2(
            model.iangle.value,
            &ldc2,
            1.0000000001 * model.phase2,
            0.0,
            1,
            model.q.value,
            model.beam_factor2.value,
            model.velocity_scale.value,
            model_beaming2,
            model.glens1,
            rlens1,
            &gint,
            star2f,
            star2c,
        );
        let fc: f64 = comp_star2(
            model.iangle.value,
            &ldc2,
            0.9999999999 * model.phase2,
            0.0,
            1,
            model.q.value,
            model.beam_factor2.value,
            model.velocity_scale.value,
            model_beaming2,
            model.glens1,
            rlens1,
            &gint,
            star2f,
            star2c,
        );
        gint.scale22 = ff / fc;
    }
    Ok(gint)
}

pub fn calc_phase(time: f64, t0: f64, period: f64, pdot: f64, delta_t: f64) -> f64 {
    let mut phase = (time - t0) / period;
    // small Newton-Raphson iteration
    for _ in 0..4 {
        phase -= (t0 + phase * (period + pdot * phase) - time)
            / (period + 2.0 * pdot * phase);
    }
    // advance/retard by time offset between primary & secondary eclipse
    phase += delta_t / period / 2.0 * ((TAU * phase).cos() - 1.0);
    phase
}

pub fn map_from_pydict(dict: Bound<'_, PyDict>) -> PyResult<HashMap<String, Entry>> {
    let mut map: HashMap<String, Entry> = HashMap::new();

    for (key, value) in dict.iter() {
        let name: String = key.extract::<String>()?;

        if let Ok(v) = value.extract::<f64>() {
            map.insert(name, Entry::Scalar(v.to_string()));
        } else if let Ok(v) = value.extract::<bool>() {
            map.insert(
                name,
                Entry::Scalar(if v { "1".to_string() } else { "0".to_string() }),
            );
        } else if let Ok(v) = value.extract::<String>() {
            map.insert(name, Entry::Scalar(v));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Unsupported type for key {}",
                name
            )));
        }
    }

    Ok(map)
}

pub fn rescale(flux: &[f64], flux_err: &[f64], weight: Option<&[f64]>, model_flux: &[f64]) -> f64 {
    let mut sdy: f64 = 0.0;
    let mut syy: f64 = 0.0;
    for i in 0..flux.len() {
        if flux_err[i] < 0.0 {
            continue;
        } else if let Some(weight) = weight {
            if weight[i] > 0.0 {
                let wgt = weight[i] / (flux_err[i] * flux_err[i]);
                sdy += wgt * flux[i] * model_flux[i];
                syy += wgt * model_flux[i] * model_flux[i];
            }
        } else {
            let wgt = 1.0 / (flux_err[i] * flux_err[i]);
            sdy += wgt * flux[i] * model_flux[i];
            syy += wgt * model_flux[i] * model_flux[i];
        }
    }
    let scale: f64 = sdy / syy;
    scale
}

pub fn chisq_log_prob(
    flux: &[f64],
    flux_err: &[f64],
    weight: Option<&[f64]>,
    model_flux: &[f64],
) -> (f64, f64) {
    let mut chisq_i: f64 = 0.0;
    let mut chisq_sum: f64 = 0.0;
    let mut log_prob: f64 = 0.0;
    for i in 0..flux.len() {
        if flux_err[i] < 0.0 {
            chisq_i = 0.0
        } else if let Some(weight) = weight {
            if weight[i] > 0.0 {
                chisq_i = weight[i] * ((flux[i] - model_flux[i]) / flux_err[i]).powi(2);
            }
        } else {
            chisq_i = ((flux[i] - model_flux[i]) / flux_err[i]).powi(2);
        }

        chisq_sum += chisq_i;
        log_prob += -0.5 * (chisq_i + (TAU * flux_err[i] * flux_err[i]).ln())
    }
    (chisq_sum, log_prob)
}

// fn apply_fluxes(points: &mut Vec<Point>, fluxes: Vec<f32>) -> Result<(), RocheError> {
//     if points.len() != fluxes.len() {
//         return Err(RocheError::ParameterError(
//             "Selected grid and flux array have mismatched lengths.".to_string(),
//         ));
//     }
//     for (point, flux) in points.iter_mut().zip(fluxes) {
//         point.set_flux(flux);
//     }
//     Ok(())
// }
