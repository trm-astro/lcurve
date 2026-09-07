use crate::ginterp::Ginterp;
use crate::grid::Grid;
use crate::ldc::LDC;
use roche::constants::C;
use roche::{self, Vec3};

//
// comp_light computes a light curve point for a particular phase. It can
// allow for finite exposures by trapezoidal integration.
// \param iangle   orbital inclination
// \param ldc1 limb darkening for star 1
// \param ldc2 limb darkening for star 2
// \param lin_limb_disc1  linear limb darkening for disc
// \param quad_limb_disc1  quadratic limb darkening for disc
// \param phase    orbital phase at centre of exposure
// \param expose   length of exposure in terms of phase
// \param ndiv     number of sub-divisions for exposure smearing using trapezoidal integration
// \param q         mass ratio
// \param beam_factor1 the 3-alpha factor for Doppler beaming for star1
// \param beam_factor2 the 3-alpha factor for Doppler beaming for star2
// \param spin1    spin/orbital ratio star 1 (makes a difference to the beaming)
// \param spin2    spin/orbital ratio star 2 (makes a difference to the beaming)
// \param vscale   the velocity scale V1+V2 (unprojected) for computation of Doppler beaming
// \param glens1   account for gravitational lensing by star 1
// \param rlens1   4x the gravitational radius (GM/c^2) of star 1, scaled by separation
// \param gint     set of numbers for switching between coarse & fine grids
// \param star1f   geometry/brightness array for star1, fine grid
// \param star2f   geometry/brightness array for star2, fine grid
// \param star1c   geometry/brightness array for star1, coarse grid
// \param star2c   geometry/brightness array for star2, coarse grid
// \param disc     geometry/brightness array for the disc
// \param edge     geometry/brightness array for the disc edge
// \param spot     geometry/brightness array for the bright spot
// \return the light curve value desired.
//

pub fn comp_light(
    iangle: f64,
    ldc1: &LDC,
    ldc2: &LDC,
    phase: f64,
    expose: f64,
    n_div: i32,
    q: f64,
    beam_factor1: f64,
    beam_factor2: f64,
    spin1: f64,
    spin2: f64,
    vscale: f64,
    glens1: bool,
    rlens1: f64,
    gint: &Ginterp,
    star1f: &Grid,
    star2f: &Grid,
    star1c: &Grid,
    star2c: &Grid,
) -> f64 {
    let x_cofm: f64 = q / (1.0 + q);
    let (sini, cosi) = iangle.to_radians().sin_cos();
    let vfac: f64 = vscale / (C / 1.0e3);

    let mut earth: Vec3;
    let mut sum = 0.0;

    let mut s: Vec3;
    let mut d: f64;
    let mut phi: f64;
    let mut p: f64;
    let mut ph: f64;
    let mut pd: f64;
    let mut phsq: f64;
    let mut wgt: f64;
    let mut vx: f64;
    let mut vy: f64;
    let mut vr: f64;
    let mut vn: f64;
    let mut mu: f64;
    let mut mud: f64;
    let mut mag: f64;
    let mut rd: f64;
    let mut ptype: i32;

    let mut ssum: f64;
    let mut ssum2: f64;

    for div in 0..n_div {
        if n_div == 1 {
            phi = phase;
            wgt = 1.0;
        } else {
            phi = phase + expose * (div as f64 - (n_div as f64 - 1.0) / 2.0) / (n_div - 1) as f64;
            if div == 0 || div == n_div - 1 {
                wgt = 0.5;
            } else {
                wgt = 1.0
            }
        }

        earth = roche::set_earth(cosi, sini, phi);

        ptype = gint.interp_type(phi);
        let star1: &Grid = if ptype == 1 { star1f } else { star1c };

        let star2: &Grid = if ptype == 3 { star2f } else { star2c };

        ssum = 0.0;

        // star 1
        for point in &star1.points {
            if point.is_visible(phi) {
                mu = earth.dot(&point.direction);
                if ldc1.see(mu) {
                    if beam_factor1 != 0.0 {
                        vx = -vfac * spin1 * point.position.y;
                        vy = vfac * (spin1 * point.position.x - x_cofm);
                        vr = -(earth.x * vx + earth.y * vy);
                        vn = point.direction.x * vx + point.direction.y * vy;
                        mud = mu - mu * vr - vn;
                        ssum +=
                            mu * (point.flux as f64) * ldc1.imu(mud) * (1.0 - beam_factor1 * vr);
                    } else {
                        ssum += mu * (point.flux as f64) * ldc1.imu(mu);
                    }
                }
            }
        }
        ssum *= gint.scale1(phi);

        // star 2
        ssum2 = 0.0;

        for point in &star2.points {
            if point.is_visible(phi) {
                mu = earth.dot(&point.direction);

                if ldc2.see(mu) {
                    // Account for magnifying effect of gravitational lensing here
                    mag = 1.0;
                    if glens1 {
                        // s = vector from centre of mass of star 1 to point of interest
                        s = point.position;

                        // d = distance along line of sight from star
                        // 1 to point in question
                        d = -s.dot(&earth);
                        if d > 0.0 {
                            // p = distance in plane of sky from cofm
                            // of star 1 to point in question pd =
                            // larger distance accounting for
                            // deflection by lensing. For p >>
                            // rlens1*d, q --> p, mag --> 1. Try to
                            // save time by avoiding square root if
                            // possible.

                            p = (s + d * earth).length();
                            ph = p / 2.0;
                            phsq = ph * ph;
                            rd = rlens1 * d;
                            if phsq > 25.0 * rd {
                                pd = p + rd / p;
                            } else {
                                pd = ph + (phsq + rd).sqrt();
                            }
                            mag = pd * pd / (pd - ph) / ph / 4.0;
                        }
                    }

                    if beam_factor2 != 0.0 {
                        vx = -vfac * spin2 * point.position.y;
                        vy = vfac * (spin2 * (point.position.x - 1.0) + 1.0 - x_cofm);
                        vr = -(earth.x * vx + earth.y * vy);
                        vn = point.direction.x * vx + point.direction.y * vy;
                        mud = mu - mu * vr - vn;
                        ssum2 += mu
                            * mag
                            * (point.flux as f64)
                            * ldc2.imu(mud)
                            * (1.0 - beam_factor2 * vr);
                    } else {
                        ssum2 += mu * mag * (point.flux as f64) * ldc2.imu(mu);
                    }
                }
            }
        }

        ssum += gint.scale2(phi) * ssum2;

        sum += wgt * ssum
    }

    sum / (1.max(n_div - 1) as f64)
}

pub fn comp_star1(
    iangle: f64,
    ldc1: &LDC,
    phase: f64,
    expose: f64,
    n_div: i32,
    q: f64,
    beam_factor1: f64,
    vscale: f64,
    model_beaming1: bool,
    gint: &Ginterp,
    star1f: &Grid,
    star1c: &Grid,
) -> f64 {
    let x_cofm: f64 = q / (1.0 + q);
    let (sini, cosi) = iangle.to_radians().sin_cos();
    let vfac: f64 = vscale / (C / 1.0e3);

    let mut earth: Vec3;
    let mut sum = 0.0;

    let mut phi: f64;
    let mut wgt: f64;
    let mut vx: f64;
    let mut vy: f64;
    let mut vr: f64;
    let mut vn: f64;
    let mut mu: f64;
    let mut mud: f64;
    let mut ptype: i32;

    let mut ssum: f64;
    let n_div_f: f64 = n_div as f64;

    for div in 0..n_div {
        if n_div == 1 {
            phi = phase;
            wgt = 1.0;
        } else {
            phi = phase + expose * (div as f64 - ((n_div_f - 1.0) / 2.0)) / (n_div_f - 1.0);

            if div == 0 || div == n_div - 1 {
                wgt = 0.5;
            } else {
                wgt = 1.0
            }
        }

        earth = roche::set_earth(cosi, sini, phi);

        // Define the grid to use
        ptype = gint.interp_type(phi);
        let star1: &Grid = if ptype == 1 { star1f } else { star1c };

        ssum = 0.0;
        let phi_normed: f64 = phi - phi.floor();
        // star 1
        for point in &star1.points {
            if point.is_visible_phase_normed(phi_normed) {
                mu = earth.dot(&point.direction);
                if ldc1.see(mu) {
                    if model_beaming1 && beam_factor1 != 0.0 {
                        vx = -vfac * point.position.y;
                        vy = vfac * (point.position.x - x_cofm);
                        vr = -(earth.x * vx + earth.y * vy);
                        vn = point.direction.x * vx + point.direction.y * vy;
                        mud = mu - mu * vr - vn;
                        ssum +=
                            mu * (point.flux as f64) * ldc1.imu(mud) * (1.0 - beam_factor1 * vr);
                    } else {
                        ssum += mu * (point.flux as f64) * ldc1.imu(mu);
                    }
                }
            }
        }
        
        sum += wgt * gint.scale1(phi) * ssum;
    }

    sum / (1.max(n_div - 1) as f64)
}

pub fn comp_star2(
    iangle: f64,
    ldc2: &LDC,
    phase: f64,
    expose: f64,
    n_div: i32,
    q: f64,
    beam_factor2: f64,
    vscale: f64,
    model_beaming2: bool,
    glens1: bool,
    rlens1: f64,
    gint: &Ginterp,
    star2f: &Grid,
    star2c: &Grid,
) -> f64 {
    let x_cofm: f64 = q / (1.0 + q);
    let (sini, cosi) = iangle.to_radians().sin_cos();
    let vfac: f64 = vscale / (C / 1.0e3);

    let mut earth: Vec3;
    let mut sum: f64 = 0.0;

    let mut s: Vec3;
    let mut d: f64;
    let mut phi: f64;
    let mut p: f64;
    let mut ph: f64;
    let mut pd: f64;
    let mut phsq: f64;
    let mut wgt: f64;
    let mut vx: f64;
    let mut vy: f64;
    let mut vr: f64;
    let mut vn: f64;
    let mut mu: f64;
    let mut mud: f64;
    let mut mag: f64;
    let mut rd: f64;
    let mut ptype: i32;

    let mut ssum: f64;
    let n_div_f: f64 = n_div as f64;

    for div in 0..n_div {
        if n_div == 1 {
            phi = phase;
            wgt = 1.0;
        } else {
            phi = phase + expose * (div as f64 - (n_div_f - 1.0) / 2.0) / (n_div_f - 1.0);
            if div == 0 || div == n_div - 1 {
                wgt = 0.5;
            } else {
                wgt = 1.0
            }
        }

        earth = roche::set_earth(cosi, sini, phi);

        // Define the grid to use
        ptype = gint.interp_type(phi);
        let star2: &Grid = if ptype == 3 { star2f } else { star2c };

        ssum = 0.0;
        let phi_normed: f64 = phi - phi.floor();
        // star 2
        for point in &star2.points {
            if point.is_visible_phase_normed(phi_normed) {
                mu = earth.dot(&point.direction);
                if ldc2.see(mu) {
                    // Account for magnifying effect of gravitational lensing here
                    mag = 1.0;
                    if glens1 {
                        // s = vector from centre of mass of star 1 to point of interest
                        s = point.position;

                        // d = distance along line of sight from star 1
                        // to point in question
                        d = -s.dot(&earth);
                        if d > 0.0 {
                            // p = distance in plane of sky from cofm
                            // of star 1 to point in question pd =
                            // larger distance accounting for
                            // deflection by lensing. For p >>
                            // rlens1*d, q --> p, mag --> 1. Try to
                            // save time by avoiding square root if
                            // possible.

                            p = (s + d * earth).length();
                            ph = p / 2.0;
                            phsq = ph * ph;
                            rd = rlens1 * d;
                            if phsq > 25.0 * rd {
                                pd = p + rd / p;
                            } else {
                                pd = ph + (phsq + rd).sqrt();
                            }
                            mag = pd * pd / (pd - ph) / ph / 4.0;
                        }
                    }

                    if model_beaming2 && beam_factor2 != 0.0  {
                        vx = -vfac * point.position.y;
                        vy = vfac * (point.position.x - x_cofm);
                        vr = -(earth.x * vx + earth.y * vy);
                        vn = point.direction.x * vx + point.direction.y * vy;
                        mud = mu - mu * vr - vn;
                        ssum += mu
                            * mag
                            * (point.flux as f64)
                            * ldc2.imu(mud)
                            * (1.0 - beam_factor2 * vr);
                    } else {
                        ssum += mu * mag * (point.flux as f64) * ldc2.imu(mu);
                    }
                }
            }
        }

        sum += wgt * gint.scale2(phi) * ssum;
    }

    sum / (1.max(n_div - 1) as f64)
}

pub fn comp_disc(
    iangle: f64,
    lin_limb_disc: f64,
    quad_limb_disc: f64,
    phase: f64,
    expose: f64,
    n_div: i32,
    disc_grid: &Grid,
) -> f64 {
    let ri = iangle.to_radians();
    let (sini, cosi) = ri.sin_cos();

    let mut earth: Vec3;
    let mut phi: f64;
    let mut sum: f64 = 0.0;
    let mut ssum;
    let mut mu;
    let mut wgt;
    let n_div_f: f64 = n_div as f64;

    for div in 0..n_div {
        if n_div == 1 {
            phi = phase;
            wgt = 1.0;
        } else {
            phi = phase + expose * (div as f64 - (n_div_f - 1.0) / 2.0) / (n_div_f - 1.0);
            if div == 0 || div == n_div - 1 {
                wgt = 0.5;
            } else {
                wgt = 1.0
            }
        }

        earth = roche::set_earth(cosi, sini, phi);

        ssum = 0.0;
        let phi_normed: f64 = phi - phi.floor();
        // Disc
        for point in &disc_grid.points {
            mu = earth.dot(&point.direction);
            if mu > 0.0 && point.is_visible_phase_normed(phi_normed) {
                ssum += mu
                    * (point.flux as f64)
                    * (1.0 - (1.0 - mu) * (lin_limb_disc + quad_limb_disc * (1.0 - mu)));
            }
        }

        sum += wgt * ssum
    }

    sum / (1.max(n_div - 1) as f64)
}

pub fn comp_disc_edge(
    iangle: f64,
    lin_limb_disc: f64,
    quad_limb_disc: f64,
    phase: f64,
    expose: f64,
    n_div: i32,
    disc_edge_grid: &Grid,
) -> f64 {
    let ri: f64 = iangle.to_radians();
    let (sini, cosi) = ri.sin_cos();

    let mut earth: Vec3;
    let mut phi: f64;
    let mut sum: f64 = 0.0;
    let mut ssum: f64;
    let mut mu: f64;
    let mut wgt: f64;
    let n_div_f: f64 = n_div as f64;

    for div in 0..n_div {
        if n_div == 1 {
            phi = phase;
            wgt = 1.0;
        } else {
            phi = phase + expose * (div as f64 - (n_div_f - 1.0) / 2.0) / (n_div_f - 1.0);
            if div == 0 || div == n_div - 1 {
                wgt = 0.5;
            } else {
                wgt = 1.0
            }
        }

        earth = roche::set_earth(cosi, sini, phi);

        ssum = 0.0;
        let phi_normed: f64 = phi - phi.floor();
        // Disc edge
        for point in &disc_edge_grid.points {
            mu = earth.dot(&point.direction);
            if mu > 0.0 && point.is_visible_phase_normed(phi_normed) {
                ssum += mu
                    * (point.flux as f64)
                    * (1.0 - (1.0 - mu) * (lin_limb_disc + quad_limb_disc * (1.0 - mu)));
            }
        }

        sum += wgt * ssum
    }

    sum / (1.max(n_div - 1) as f64)
}

pub fn comp_bright_spot(
    iangle: f64,
    phase: f64,
    expose: f64,
    n_div: i32,
    bright_spot_grid: &Grid,
) -> f64 {
    let ri = iangle.to_radians();
    let (sini, cosi) = ri.sin_cos();

    let mut earth: Vec3;
    let mut phi: f64;
    let mut sum: f64 = 0.0;
    let mut ssum;
    let mut mu;
    let mut wgt;
    let n_div_f: f64 = n_div as f64;

    for div in 0..n_div {
        if n_div == 1 {
            phi = phase;
            wgt = 1.0;
        } else {
            phi = phase + expose * (div as f64 - (n_div_f - 1.0) / 2.0) / (n_div_f - 1.0);
            if div == 0 || div == n_div - 1 {
                wgt = 0.5;
            } else {
                wgt = 1.0
            }
        }

        earth = roche::set_earth(cosi, sini, phi);

        ssum = 0.0;
        // Bright spot
        for point in &bright_spot_grid.points {
            mu = earth.dot(&point.direction);
            if mu > 0.0 && point.is_visible(phi) {
                ssum += mu * (point.flux as f64);
            }
        }

        sum += wgt * ssum
    }

    sum / (1.max(n_div - 1) as f64)
}
