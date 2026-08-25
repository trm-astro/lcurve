use crate::model::Model;
use crate::set_star_grid::star_eclipse;
use roche::{self, Etype, Point, RocheContext, Star, Vec3, errors::RocheError};
use std::f64::consts::FRAC_PI_2;
use std::panic;

//
// set_bright_spot_grid sets up the elements needed to define the bright spot
// at the edge of the disc. This is modelled as two straight lines of elements
// coincident in position, but with one oriented parallel to the plane of the
// disc and the other tilted by an arbitrary angle. This is to allow the
// elements to make a hump while allowing flexibility in the step heights of
// the bright-spot ingress and egress features. The brightness of the spot
// along the lines is modelled as a function of distance x from the start of
// each line as (x/l)**p1*exp(-(x/l)**p2) where l is a scale length and p1 and
// p2 are power law exponents. The maximum of this occurs at x =
// l*(p1/p2)**(1/p2), and is set to be the intersection of the gas stream with
// the defined spot radius from the white dwarf.
//
// \param mdl       Model object defining parameters
// \param spot      array representing the spot
// \exception Exceptions are thrown if the specified radii over-fill the Roche lobes.
//

pub fn set_bright_spot_grid(model: &Model) -> Result<Vec<Point>, RocheError> {
    let (mut r1, mut r2) = model.get_r1r2();

    let roche_context1 = RocheContext::new(model.q.value, Star::Primary, model.spin1.value)?;
    let roche_context2 = RocheContext::new(model.q.value, Star::Secondary, model.spin2.value)?;

    let rl1: f64 = roche_context1.x_l1_asyncronous()?;
    if r1 < 0.0 {
        r1 = rl1;
    } else if r1 > rl1 {
        panic!("Primary star is larger than its Roche Lobe!");
    }

    let rl2: f64 = 1.0 - roche_context2.x_l1_asyncronous()?;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        panic!("Secondary star is larger than its Roche Lobe!");
    }

    let ffac1: f64 = r1 / rl1;
    let ffac2: f64 = r2 / rl2;

    let (mut bspot, mut v) = roche::strinit(model.q.value)?;
    roche::stradv(
        model.q.value,
        &mut bspot,
        &mut v,
        model.radius_spot.value,
        1.0e-10,
        1.0e-3,
    )?;

    // Now measure bright-spot angle relative to tangent to disc edge so we need
    // to add 90 + angle of bright-spot to the input value.
    let theta: f64 = bspot.y.atan2(bspot.x) + FRAC_PI_2 + model.angle_spot.value.to_radians();

    let (sin_theta, cos_theta) = theta.sin_cos();

    let alpha: f64 = model.yaw_spot.value.to_radians();
    let tilt: f64 = model.tilt_spot.value.to_radians();
    let (sin_tilt, cos_tilt) = tilt.sin_cos();

    // The direction of the line of elements is set by angle_spot, but the
    // beaming direction adds in yaw_spot as well.

    let bvec: Vec3 = Vec3::new(cos_theta, sin_theta, 0.0);
    let pvec: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    let tvec: Vec3 = Vec3::new(
        sin_tilt * (theta + alpha).sin(),
        -sin_tilt * (theta + alpha).cos(),
        cos_tilt,
    );

    let b_max: f64 =
        (model.expon_spot.value / model.epow_spot.value).powf(1. / model.epow_spot.value);
    let sfac: f64 = 20. + b_max;

    let mut bright_spot_grid: Vec<Point> = Vec::with_capacity(2 * model.nspot as usize);
    bright_spot_grid.resize(2 * model.nspot as usize, Point::default());

    let mut eclipses = Etype::new();

    // This is where the spot height gets in
    let area: f64 =
        sfac * model.length_spot.value * model.height_spot.value / (model.nspot as f64 - 1.0);
    let bright: f64 = roche::planck(model.wavelength, model.temp_spot.value);

    for i in 0..model.nspot as usize {
        // Position is adjusted to locate the impact point at the peak temperature point
        let dist: f64 = sfac * i as f64 / (model.nspot - 1) as f64;
        let posn: Vec3 = bspot + model.length_spot.value * (dist - b_max) * bvec;

        eclipses.clear();
        if model.eclipse1 {
            star_eclipse(
                &roche_context1,
                r1,
                ffac1,
                model.iangle.value,
                &posn,
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
                model.iangle.value,
                &posn,
                model.delta_phase,
                model.roche2,
                Star::Secondary,
                &mut eclipses,
            )?;
        }

        // Factor here is adjusted to equal 1 at its peak
        let modified_brightness: f64 = bright
            * (dist / b_max).powf(model.expon_spot.value)
            * (((model.expon_spot.value / model.epow_spot.value)
                - dist.powf(model.epow_spot.value))
            .exp());

        let flux_tilt: f32 = (modified_brightness * (1.0 - model.cfrac_spot.value) * (area as f32) as f64) as f32;
        let flux_parallel: f32 = (modified_brightness * model.cfrac_spot.value * (area as f32) as f64) as f32;
        // the tilted strip
        bright_spot_grid[i] = Point {
            position: posn,
            direction: tvec,
            area: area as f32,
            gravity: 1.0,
            eclipse: eclipses.clone(),
            flux: flux_tilt,
        };

        // the parallel strip
        bright_spot_grid[model.nspot as usize + i] = Point {
            position: posn,
            direction: pvec,
            area: area as f32,
            gravity: 1.0,
            eclipse: eclipses.clone(),
            flux: flux_parallel,
        };
    }
    Ok(bright_spot_grid)
}
