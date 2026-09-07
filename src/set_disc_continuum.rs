use roche::{self, Vec3};
use std::f64::consts::{FRAC_PI_2, PI};

use crate::grid::Grid;

// set_disc_continuum computes the face-on brightness of each element of the
// disc assuming a power law with radius.
//
// \param rdisc      radius of disc where the next parameter is defined
// \param tdisc      the temperature at radius rdisc in the disc. This is
//                   used to set the surface brightness scale.
// \param texp       the exponent controlling the scaling of brightness with
//                   radius. Negative to increase towards the centre of the
//                   disc.
// \param wave       wavelength of interest, nm
// \param disc       grid of elements over disc

pub fn set_disc_continuum(rdisc: f64, tdisc: f64, texp: f64, wave: f64, disc: &mut Grid) {
    // Reference surface brightness
    let bright: f64 = roche::planck(wave, tdisc);

    for point in &mut disc.points {
        let r: f64 = point.position.length();
        point.flux = (bright * (r / rdisc).powf(texp) * point.area as f64) as f32;
    }
}

// set_edge_continuum computes the face-on brightness of each element of the
// edge of the disc assuming it intrinsically has temperature tedge
// plus whatever it gets from irradiation. This to allow for unsual
// sdO+WD accreting system found by Thomas Kupfer. Donor approximated
// as point source of luminosity defined by its radius and temperature.
//
// \param tedge      temperature at outer edge of disc
// \param r2         radius of donor
// \param t2         temperature of donor
// \param absorb     amount of irradiation flux absorbed and reprocessed
// \param wave       wavelength of interest, nm
// \param edge       grid of elements over disc
//

pub fn set_edge_continuum(
    tedge: f64,
    r2: f64,
    t2: f64,
    absorb: f64,
    wave: f64,
    edge: &mut Grid,
) {
    let mut vec: Vec3;
    let cofm2: Vec3 = Vec3::cofm2();
    let mut temp: f64;
    let mut geom: f64;
    let mut mu: f64;
    let mut r: f64;

    for point in &mut edge.points {
        vec = cofm2 - point.position;
        r = vec.length();
        mu = vec.dot(&point.direction) / r;

        if mu >= r2 {
            // Full tilt irradiation
            geom = (r2 / r) * (r2 / r) * mu;
            temp = (tedge.powi(4) + absorb * geom * t2.powi(4)).powf(0.25);
        } else if mu > -r2 {
            // 'sunset' case
            let x0: f64 = -mu / r2;

            // The following factor is a weighted version of 'mu' as the
            // secondary sets as far as this element is concerned.  When x0 =
            // -1 it equals r2 = mu. when x0 = 0 it equals 2*r2/(3*Pi) as
            // opposed to zero which it would be in the point source case.
            geom = (r2 / r)
                * (r2 / r)
                * r2
                * ((1.0 - x0 * x0).sqrt() * (2.0 + x0 * x0) / 3.0 - x0 * (FRAC_PI_2 - x0.asin()))
                / PI;
            temp = (tedge.powi(4) + absorb * geom * t2.powi(4)).powf(0.25);
        } else {
            // No irradiation
            temp = tedge;
        }
        point.set_flux(((point.area as f64) * roche::planck(wave, temp)) as f32);
    }
}
