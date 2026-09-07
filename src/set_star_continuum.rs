use crate::{grid::Grid, model::Model};
use roche::{self, Vec3, constants::EFAC, errors::RocheError};
use std::f64::consts::PI;

pub fn set_star_continuum(
    model: &Model,
    star1: &mut Grid,
    star2: &mut Grid,
) -> Result<(), RocheError> {
    let (mut r1, mut r2) = model.get_r1r2();

    let rl1: f64 = roche::x_l1(model.q.value)?;
    if r1 < 0.0 {
        r1 = rl1;
    }
    let rl2: f64 = 1.0 - rl1;
    if r2 < 0.0 {
        r2 = rl2;
    }
    let cofm2 = Vec3::cofm2();

    // First compute irradiation of star 1 by star 2. 'geom' is the
    // geometrical factor by which the radiation from star 2 is reduced
    // by the time it hits the element in terms of flux per unit area
    // compared to its level as it leaves star 2
    let mut geom: f64;
    let mut temp: f64;

    // modify the gravity darkening coefficient to allow for two possibilities
    // the gravity darkening is implemented by modifying the temperature and
    // then calculating the flux in a BB approx. The 'bolometric' method does
    // this directly; the 'filter integrated' method modifies the exponent
    // used to give the desired behaviour of flux with gravity.
    let gdcbol1: f64 = if model.gdark_bolom1 {
        model.gravity_dark1.value
    } else {
        model.gravity_dark1.value / roche::dlpdlt(model.wavelength, model.t1.value)
    };

    // compute direction of star spot 11, 12, 13, and the uespot
    let is_spot11: bool = model.stsp11_long.defined
        && model.stsp11_lat.defined
        && model.stsp11_fwhm.defined
        && model.stsp11_tcen.defined;
    let is_spot12: bool = model.stsp12_long.defined
        && model.stsp12_lat.defined
        && model.stsp12_fwhm.defined
        && model.stsp12_tcen.defined;
    let is_spot13: bool = model.stsp13_long.defined
        && model.stsp13_lat.defined
        && model.stsp13_fwhm.defined
        && model.stsp13_tcen.defined;
    let is_uespot: bool = model.uesp_long1.defined
        && model.uesp_long2.defined
        && model.uesp_lathw.defined
        && model.uesp_taper.defined
        && model.uesp_temp.defined;

    let spot11: Vec3 = if is_spot11 {
        let (slong11, clong11) = model.stsp11_long.value.to_radians().sin_cos();
        let (slat11, clat11) = model.stsp11_lat.value.to_radians().sin_cos();
        Vec3::new(clat11 * clong11, clat11 * slong11, slat11)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };

    let spot12: Vec3 = if is_spot12 {
        let (slong12, clong12) = model.stsp12_long.value.to_radians().sin_cos();
        let (slat12, clat12) = model.stsp12_lat.value.to_radians().sin_cos();
        Vec3::new(clat12 * clong12, clat12 * slong12, slat12)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };

    let spot13: Vec3 = if is_spot13 {
        let (slong13, clong13) = model.stsp13_long.value.to_radians().sin_cos();
        let (slat13, clat13) = model.stsp13_lat.value.to_radians().sin_cos();
        Vec3::new(clat13 * clong13, clat13 * slong13, slat13)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };

    let mut longhw: f64 = 0.0;
    let uespot: Vec3 = if is_uespot {
        longhw = (model.uesp_long2.value - model.uesp_long1.value) / 2.0;
        let lcen: f64 = ((model.uesp_long1.value + model.uesp_long2.value) / 2.0).to_radians();
        let (slong, clong) = lcen.sin_cos();
        Vec3::new(clong, slong, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };

    for point in &mut star1.points {
        let vec: Vec3 = cofm2 - point.position;
        let r: f64 = vec.length();
        let mu: f64 = point.direction.dot(&vec) / r;

        // compute unirradiated temperature allowing for
        // offset from spot centre
        let mut t1: f64 = model.t1.value;

        if is_spot11 {
            let dist: f64 = (spot11.dot(&point.position) / point.position.length())
                .acos()
                .to_degrees();
            let exponent: f64 = -(dist / (model.stsp11_fwhm.value / EFAC)).powi(2) / 2.0;
            t1 += (model.stsp11_tcen.value - model.t1.value) * exponent.exp();
        }

        if is_spot12 {
            let dist: f64 = (spot12.dot(&point.position) / point.position.length())
                .acos()
                .to_degrees();
            let exponent: f64 = -(dist / (model.stsp12_fwhm.value / EFAC)).powi(2) / 2.0;
            t1 += (model.stsp12_tcen.value - model.t1.value) * exponent.exp();
        }

        if is_spot13 {
            let dist: f64 = (spot13.dot(&point.position) / point.position.length())
                .acos()
                .to_degrees();
            let exponent: f64 = -(dist / (model.stsp13_fwhm.value / EFAC)).powi(2) / 2.0;
            t1 += (model.stsp13_tcen.value - model.t1.value) * exponent.exp();
        }

        if is_uespot {
            // equatorial spot
            let evec: Vec3 = Vec3::new(point.position.x, point.position.y, 0.0);
            let evec_length = evec.length();

            if evec_length > 0.0 {
                // compute latitude and longitude offset
                let rd: f64 = point.position.length();
                let mut clat: f64 = evec.dot(&point.position) / evec_length / rd;
                clat = 1.0_f64.min(-1.0_f64.max(clat));
                let dlat: f64 = clat.acos().to_degrees();

                let mut clong: f64 = uespot.dot(&evec) / evec_length;
                clong = 1.0_f64.min(-1.0_f64.max(clong));
                let dlong: f64 = clong.acos().to_degrees();

                if dlat <= model.uesp_lathw.value && dlong <= longhw {
                    t1 = model.uesp_temp.value
                } else if dlat <= model.uesp_lathw.value {
                    t1 += (model.uesp_temp.value - model.t1.value)
                        * (-(dlat - model.uesp_lathw.value) / model.uesp_taper.value).exp();
                } else {
                    t1 += (model.uesp_temp.value - model.t1.value)
                        * (-(dlat - model.uesp_lathw.value) / model.uesp_taper.value).exp()
                        * (-(dlong - longhw) / model.uesp_taper.value).exp();
                }
            }
        }

        if mu >= r2 {
            // Full tilt irradiation
            geom = (r2 / r) * (r2 / r) * mu;
            temp = ((t1 * (point.gravity as f64).powf(gdcbol1)).powi(4)
                + model.absorb.value * model.t2.value.powi(4) * geom)
                .powf(0.25);
        } else if mu > -r2 {
            // 'sunset' case
            let x0: f64 = -mu / r2;
            // The following factor is a weighted version of 'mu' as the
            // secondary sets as far as this element is concerned.  When x0 =
            // -1 it equals r2 = mu. when x0 = 0 it equals 2*r2/(3*Pi) as
            // opposed to zero which it would be in the point source case
            geom = (r2 / r)
                * (r2 / r)
                * r2
                * ((1.0 - x0 * x0).sqrt() * (2.0 + x0 * x0) / 3.0 - x0 * (PI / 2.0 - x0.asin()))
                / PI;

            temp = ((t1 * (point.gravity as f64).powf(gdcbol1)).powi(4)
                + model.absorb.value * model.t2.value.powi(4) * geom)
                .powf(0.25);
        } else {
            // No irradiation
            geom = 0.0;
            temp = t1 * (point.gravity as f64).powf(gdcbol1);
        }

        // At this stage also add in a directly reflected part too
        let mut flux: f32 = ((point.area as f64) * roche::planck(model.wavelength, temp)) as f32;

        if model.mirror {
            flux += point.area
                * (geom as f32)
                * (roche::planck(model.wavelength, model.t2.value.abs()) as f32);
        }

        point.set_flux(flux);
    }

    let cofm1 = Vec3::cofm1();

    // See comments on GDCBOL1
    let gdcbol2: f64 = if model.gdark_bolom2 {
        model.gravity_dark2.value
    } else {
        model.gravity_dark2.value / roche::dlpdlt(model.wavelength, model.t2.value.abs())
    };

    let is_spot21: bool = model.stsp21_long.defined
        && model.stsp21_lat.defined
        && model.stsp21_fwhm.defined
        && model.stsp21_tcen.defined;
    let is_spot22: bool = model.stsp22_long.defined
        && model.stsp22_lat.defined
        && model.stsp22_fwhm.defined
        && model.stsp22_tcen.defined;

    let spot21: Vec3 = if is_spot21 {
        let (slong21, clong21) = model.stsp21_long.value.to_radians().sin_cos();
        let (slat21, clat21) = model.stsp21_lat.value.to_radians().sin_cos();
        Vec3::new(clat21 * clong21, clat21 * slong21, slat21)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };

    let spot22: Vec3 = if is_spot22 {
        let (slong22, clong22) = model.stsp22_long.value.to_radians().sin_cos();
        let (slat22, clat22) = model.stsp22_lat.value.to_radians().sin_cos();
        Vec3::new(clat22 * clong22, clat22 * slong22, slat22)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };

    for point in &mut star2.points {
        let vec: Vec3 = cofm1 - point.position;
        let r: f64 = vec.length();
        let mu: f64 = point.direction.dot(&vec) / r;

        // compute unirradiated temperature allowing for
        // offset from spot centre
        let mut t2 = model.t2.value.abs();

        if is_spot21 {
            let off: Vec3 = point.position - cofm2;
            let dist: f64 = (spot21.dot(&off) / off.length()).acos().to_degrees();
            let exponent: f64 = -(dist / (model.stsp11_fwhm.value / EFAC)).powi(2) / 2.0;
            t2 += (model.stsp21_tcen.value - t2) * exponent.exp();
        }

        if is_spot22 {
            let off: Vec3 = point.position - cofm2;
            let dist: f64 = (spot22.dot(&off) / off.length()).acos().to_degrees();
            let exponent: f64 = -(dist / (model.stsp11_fwhm.value / EFAC)).powi(2) / 2.0;
            t2 += (model.stsp22_tcen.value - t2) * exponent.exp();
        }

        if mu >= r1 {
            // Full tilt irradiation
            geom = (r1 / r) * (r1 / r) * mu;
            temp = ((t2 * (point.gravity as f64).powf(gdcbol2)).powi(4)
                + model.absorb.value * model.t1.value.powi(4) * geom)
                .powf(0.25);
        } else if mu > -r1 {
            // 'sunset' case
            let x0: f64 = -mu / r1;
            geom = (r1 / r)
                * (r1 / r)
                * r1
                * ((1.0 - x0 * x0).sqrt() * (2.0 + x0 * x0) / 3.0 - x0 * (PI / 2.0 - x0.asin()))
                / PI;

            temp = ((t2 * (point.gravity as f64).powf(gdcbol2)).powi(4)
                + model.absorb.value * model.t1.value.powi(4) * geom)
                .powf(0.25);
        } else {
            // No irradiation
            geom = 0.0;
            temp = t2 * (point.gravity as f64).powf(gdcbol2);
        }

        // At this stage also add in a directly reflected part too
        let mut flux: f32 = ((point.area as f64) * roche::planck(model.wavelength, temp)) as f32;
        if model.mirror {
            flux += point.area
                * (geom as f32)
                * (roche::planck(model.wavelength, model.t1.value) as f32);
        }

        point.set_flux(flux);
    }
    Ok(())
}
