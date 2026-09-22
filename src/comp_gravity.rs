use crate::grid::Grid;
use crate::model::Model;
use roche::constants::DAY;
use roche::errors::RocheError;
use roche::{RocheContext, Star, Vec3};
use std::f64::consts::TAU;

//
//  comp_gravity1 computes a flux-weighted gravity for star1 as an approximate
//  estimate of its gravity. Returned as logg in cgs units, directly comparable
//  to usual surface gravity. Just uses fine grid.
//
//  \param mdl -- the current model
//  \param star1 -- grid for star1
//  \return the value of logg
//

pub fn comp_gravity1(model: &Model, star1_fine_grid: &Grid) -> Result<f64, RocheError> {
    // Calculate the unit scaling factor to get CGS gravity
    let gm1m2: f64 = (1000.0 * model.velocity_scale.value).powi(3) * model.tperiod * DAY / TAU;
    let a: f64 = (gm1m2 / (TAU / DAY / model.tperiod).powi(2)).powf(1.0 / 3.0);
    let gscale: f64 = 100.0 * gm1m2 / (a * a);
    let mut gref: f64;

    // radius stuff
    let (r1, _) = model.get_r1r2();
    let rochecontext1: RocheContext =
        RocheContext::new(model.q.value, Star::Primary, model.spin1.value)?;
    let rl1: f64 = rochecontext1.x_l1;

    // calculate the reference gravity in CGS units
    if model.roche1 {
        let acc: f64 = model.delta_phase / 10.0;
        let ffac1: f64 = r1 / rl1;
        let (rref1, pref1) = rochecontext1.ref_sphere(ffac1)?;

        let dirn = Vec3::new(-1.0, 0.0, 0.0);
        (_, _, _, gref) = rochecontext1.face(dirn, rref1, pref1, acc)?;
        gref *= gscale;
    } else {
        gref = gscale / (1.0 + model.q.value) / (r1 * r1);
    }

    let mut sumfg: f64 = 0.0;
    let mut sumf: f64 = 0.0;

    // Star 1
    for point in &star1_fine_grid.points {
        // flux has built-in area factor
        sumfg += (point.flux * point.gravity) as f64;
        sumf += point.flux as f64;
    }
    if (gref > 0.0) & (sumfg > 0.0) & (sumf > 0.0) {
        Ok((gref * sumfg / sumf).log10())
    } else {
        Ok(0.0)
    }
}

//
//  comp_gravity2 computes a flux-weighted gravity for star2 as an approximate
//  estimate of its gravity. Returned as logg in cgs units, directly comparable
//  to usual surface gravity. Just uses fine grid.
//
//  \param mdl -- the current model
//  \param star2 -- grid for star2
//  \return the value of logg
//

pub fn comp_gravity2(model: &Model, star2_fine_grid: &Grid) -> Result<f64, RocheError> {
    // Calculate the unit scaling factor to get CGS gravity
    let gm1m2: f64 = (1000.0 * model.velocity_scale.value).powi(3) * model.tperiod * DAY / TAU;
    let a: f64 = (gm1m2 / (TAU / DAY / model.tperiod).powi(2)).powf(1.0 / 3.0);
    let gscale: f64 = 100.0 * gm1m2 / (a * a);
    let mut gref: f64;

    // radius stuff
    let (_, mut r2) = model.get_r1r2();
    let rochecontext2: RocheContext =
        RocheContext::new(model.q.value, Star::Secondary, model.spin1.value)?;
    let rl2: f64 = 1.0 - rochecontext2.x_l1;
    if model.r2.value < 0.0 {
        r2 = rl2;
    }

    // calculate the reference gravity in CGS units
    if model.roche2 {
        let acc: f64 = model.delta_phase / 10.0;
        let ffac2: f64 = r2 / rl2;
        let (rref2, pref2) = rochecontext2.ref_sphere(ffac2)?;

        let dirn = Vec3::new(1.0, 0.0, 0.0);
        (_, _, _, gref) = rochecontext2.face(dirn, rref2, pref2, acc)?;
        gref *= gscale;
    } else {
        gref = gscale * model.q.value / (1.0 + model.q.value) / (r2 * r2);
    }

    let mut sumfg: f64 = 0.0;
    let mut sumf: f64 = 0.0;

    // Star 2
    for point in &star2_fine_grid.points {
        // flux has built-in area factor
        sumfg += (point.flux * point.gravity) as f64;
        sumf += point.flux as f64;
    }
    if (gref > 0.0) & (sumfg > 0.0) & (sumf > 0.0) {
        Ok((gref * sumfg / sumf).log10())
    } else {
        Ok(0.0)
    }
}
