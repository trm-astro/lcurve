use crate::model::Model;
use crate::numface::numface;
use rayon::prelude::*;
use roche::constants;
use roche::errors::RocheError;
use roche::{self, Etype, Point, RocheContext, Star, Vec3};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::panic;

pub struct Xy {
    pub x: f64,
    pub y: f64,
}

pub fn envelope(rangle: f64, lambda: f64, r1: f64) -> Xy {
    let (sini, cosi) = rangle.sin_cos();
    let (sinl, cosl) = lambda.sin_cos();
    let norm: f64 = (cosi * cosi + sini * sini * cosl * cosl).sqrt();
    Xy {
        x: (sinl + r1 * cosi * sinl / norm),
        y: (-cosi * cosl - r1 * cosl / norm),
    }
}

pub fn set_star_grid(model: &Model, star: Star, fine: bool) -> Result<Vec<Point>, RocheError> {
    let (mut r1, mut r2) = model.get_r1r2();

    let eclipse: bool = match star {
        Star::Primary => model.eclipse1,
        Star::Secondary => model.eclipse2,
    };
    let nlat: u32 = match (star, fine) {
        (Star::Primary, true) => model.nlat1f,
        (Star::Primary, false) => model.nlat1c,
        (Star::Secondary, true) => model.nlat2f,
        (Star::Secondary, false) => model.nlat2c,
    };
    let nlatfill: u32 = match (star, fine) {
        (Star::Primary, true) => 0,
        (Star::Primary, false) => 0,
        (Star::Secondary, false) => 0,
        (Star::Secondary, true) => model.nlatfill,
    };
    let nlngfill: u32 = match (star, fine) {
        (Star::Primary, true) => 0,
        (Star::Primary, false) => 0,
        (Star::Secondary, false) => 0,
        (Star::Secondary, true) => model.nlngfill,
    };
    let latlo: f64 = match star {
        Star::Primary => 0.0,
        Star::Secondary => model.llo,
    };
    let lathi: f64 = match star {
        Star::Primary => 0.0,
        Star::Secondary => model.lhi,
    };

    let roche_context1 = RocheContext::new(model.q.value, Star::Primary, model.spin1.value)?;
    let roche_context2 = RocheContext::new(model.q.value, Star::Secondary, model.spin2.value)?;

    let rl1: f64 = roche_context1.x_l1;
    if r1 < 0.0 {
        r1 = rl1;
    } else if r1 > rl1 {
        panic!("set_star_grid: the primary star is larger than its Roche lobe!");
    }

    let rl2: f64 = 1.0 - roche_context2.x_l1;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        panic!("set_star_grid: the secondary star is larger than its Roche lobe!");
    }

    if model.glens1 && star == Star::Secondary && model.roche1 && model.eclipse2 {
        panic!(
            "set_star_grid: cannot have gravitational lensing, eclipse and Roche lobe geometry at the same time"
        );
    }

    // Gravitational lensing accounted for here by shrinking the lensing star
    // by a fixed amount when it is not the one having its grid computed. This
    // enables a proper computation of the eclipse of the star having its grid
    // computed by its companion. The proper correction to the radius is: r
    // --> r - 4GMd/(c^2 r) where d is the distance from the limb of the other
    // star to the point in question Here it is assumed that d = a - R/2, the
    // orbital separation minus half the radius of the star being eclipse,
    // which is approximate but should not be too bad.

    if model.glens1 && star == Star::Secondary {
        let gm1m2 =
            (1000.0 * model.velocity_scale.value).powi(3) * model.tperiod * constants::DAY / TAU;
        let a: f64 = (gm1m2 / (TAU / constants::DAY / model.tperiod).powi(2)).powf(1.0 / 3.0);
        let rlens1: f64 = 4.0 * gm1m2 / (1.0 + model.q.value) / a / constants::C.powi(2);

        r1 -= rlens1 / (r1 / (1.0 - r2 / 2.0));
        if r1 < 0.0 {
            panic!(
                "set_star_grid: gravitational lensing correction more than the current program can cope with."
            );
        }
    }

    // Calculate a reference radius and potential for the two stars
    let ffac1: f64 = r1 / rl1;
    let (rref1, pref1) = roche_context1.ref_sphere(ffac1)?;

    let ffac2: f64 = r2 / rl2;
    let (rref2, pref2) = roche_context2.ref_sphere(ffac2)?;

    // Compute latitude range over which extra points will be added. Only enabled
    // when setting the secondary grid and when the grid North pole is the genuine
    // North pole and when r2 > r1

    let mut infill: bool =
        model.npole && (star == Star::Secondary) && (nlatfill > 0 || nlngfill > 0) && r2 > r1;
    let mut thelo: f64 = 0.0;
    let mut thehi: f64 = 0.0;
    if infill {
        let rangle: f64 = model.iangle.value.to_radians();
        let cosi: f64 = rangle.cos();
        let mut ratio = (cosi + r1) / r2;

        // Lower latitude value comes from assuming star 1 is seen
        // at the limb of star 2
        if ratio >= 1.0 {
            thehi = FRAC_PI_2 + rangle;
        } else {
            // binary chop to discover where lower envelope of star 1 crosses edge of star 2
            let mut llo: f64 = 0.0;
            let mut lhi: f64 = FRAC_PI_2;
            let mut xy: Xy = Xy { x: 0.0, y: 0.0 };

            while lhi > llo + 1.0e-7 {
                let lmid: f64 = (llo + lhi) / 2.0;
                xy = envelope(rangle, lmid, r1);
                if xy.x * xy.x + xy.y * xy.y < r2 * r2 {
                    llo = lmid;
                } else {
                    lhi = lmid;
                }
            }
            let sini: f64 = rangle.sin();

            // Final value, apply the lower latitude limit
            thehi = (FRAC_PI_2 - latlo.to_radians()).max(
                (FRAC_PI_2 + rangle).min((xy.y * sini / r2).acos() + model.lfudge.to_radians()),
            );
        }

        // Upper latitude value comes from uppermost latitude covered when star 1 crosses
        // meridian of star 2.
        ratio = (cosi - r1) / r2;
        if ratio >= 1.0 {
            infill = false;
        } else if ratio <= -1.0 {
            thelo = 0.0;
        } else {
            thelo = (FRAC_PI_2 - lathi.to_radians())
                .min(0.0f64.max(FRAC_PI_2 - ratio.acos() + rangle - model.lfudge.to_radians()))
        }
    }

    // Compute number of faces needed
    let nface: u32 = numface(nlat, infill, thelo, thehi, nlatfill, nlngfill);

    // Generate arrays over the star's face
    let mut star_grid: Vec<Point> = Vec::with_capacity(nface as usize);

    let acc: f64 = model.delta_phase / 10.0;

    // let mut dirn: Vec3;
    let _posn: Vec3;
    let _dvec: Vec3;
    let _rad: f64;
    let gref: f64;
    // Compute reference gravity value, from the side of the star opposite from the L1 point
    // to ensure a non-zero value. Set to 1 if Roche distortion being ignored.
    if star == Star::Primary && model.roche1 {
        let dirn = Vec3::new(-1.0, 0.0, 0.0);
        (_posn, _dvec, _rad, gref) = roche_context1.face(dirn, rref1, pref1, acc)?;
    } else if star == Star::Secondary && model.roche2 {
        let dirn = Vec3::new(1.0, 0.0, 0.0);
        (_posn, _dvec, _rad, gref) = roche_context2.face(dirn, rref2, pref2, acc)?;
    } else {
        gref = 1.0;
    }

    // The grid starts at the North pole and ends at the South, proceeding in
    // a series of equi-latitudinal rings.  The polar axis is parallel to the
    // x-axis which points from one star to the other. The North pole is
    // defined to be the point closest to the other star (or the genuine North
    // Pole if npole is true). The angle theta is measured away from the North
    // pole, the angle phi is measured from the Y axis towards the Z axis.

    let dtheta: f64 = PI / (nlat as f64);

    if infill {
        add_faces(
            &mut star_grid,
            0.0,
            thelo,
            dtheta,
            0,
            0,
            model.npole,
            star,
            &roche_context1,
            &roche_context2,
            model.iangle.value,
            r1,
            r2,
            rref1,
            rref2,
            model.roche1,
            model.roche2,
            eclipse,
            gref,
            pref1,
            pref2,
            ffac1,
            ffac2,
            model.delta_phase,
        );
        add_faces(
            &mut star_grid,
            thelo,
            thehi,
            dtheta,
            nlatfill,
            nlngfill,
            model.npole,
            star,
            &roche_context1,
            &roche_context2,
            model.iangle.value,
            r1,
            r2,
            rref1,
            rref2,
            model.roche1,
            model.roche2,
            eclipse,
            gref,
            pref1,
            pref2,
            ffac1,
            ffac2,
            model.delta_phase,
        );
        add_faces(
            &mut star_grid,
            thehi,
            PI,
            dtheta,
            0,
            0,
            model.npole,
            star,
            &roche_context1,
            &roche_context2,
            model.iangle.value,
            r1,
            r2,
            rref1,
            rref2,
            model.roche1,
            model.roche2,
            eclipse,
            gref,
            pref1,
            pref2,
            ffac1,
            ffac2,
            model.delta_phase,
        );
    } else {
        add_faces(
            &mut star_grid,
            0.0,
            PI,
            dtheta,
            0,
            0,
            model.npole,
            star,
            &roche_context1,
            &roche_context2,
            model.iangle.value,
            r1,
            r2,
            rref1,
            rref2,
            model.roche1,
            model.roche2,
            eclipse,
            gref,
            pref1,
            pref2,
            ffac1,
            ffac2,
            model.delta_phase,
        );
    }

    Ok(star_grid)
}

pub fn add_faces(
    star_grid: &mut Vec<Point>,
    tlo: f64,
    thi: f64,
    dtheta: f64,
    nlatfill: u32,
    nlngfill: u32,
    npole: bool,
    star: Star,
    roche_context1: &RocheContext,
    roche_context2: &RocheContext,
    iangle: f64,
    r1: f64,
    r2: f64,
    rref1: f64,
    rref2: f64,
    roche1: bool,
    roche2: bool,
    eclipse: bool,
    gref: f64,
    pref1: f64,
    pref2: f64,
    ffac1: f64,
    ffac2: f64,
    delta: f64,
) {
    let ri: f64 = iangle.to_radians();
    let (sini, cosi) = ri.sin_cos();

    // Centre of masses of the stars
    let cofm1 = Vec3::cofm1();
    let cofm2 = Vec3::cofm2();

    // Can afford to be pretty careful on the location of faces as it is a fast computation
    let acc: f64 = delta / 10.0;

    let infill: bool = (nlatfill > 0) || (nlngfill > 0);

    let nlat: usize;
    let mut nlat1: usize = 0;
    let mut nlat2: usize = 0;

    if infill {
        // If infill is True we loop through the latitudes once on the side
        // facing the other star and once on the other side so that infilling
        // only occurs on one side. The infilled part will be included first
        nlat1 = ((1.0 + nlatfill as f64) * (thi - tlo) / dtheta).ceil() as usize;
        nlat2 = ((thi - tlo) / dtheta).ceil() as usize;
        nlat = nlat1 + nlat2;
    } else {
        nlat = ((thi - tlo) / dtheta).ceil() as usize;
    };

    let bands: Vec<Vec<Point>> = (0..nlat)
        .into_par_iter()
        .map(|nt| {
            let g = band_geometry(
                nt, tlo, thi, dtheta, nlat, nlat1, nlat2, infill, nlngfill, star,
            );

            let mut band = Vec::with_capacity(g.nphi);

            for np in 0..g.nphi {
                let phi = g.phi1 + (g.phi2 - g.phi1) * (np as f64 + 0.5) / g.nphi as f64;

                let (sinp, cosp) = phi.sin_cos();

                let mut dirn = Vec3::new(0.0, 0.0, 0.0);
                if npole {
                    dirn.set(g.sint * cosp, g.sint * sinp, g.cost);
                } else {
                    dirn.set(g.cost, g.sint * cosp, g.sint * sinp);
                }

                let posn: Vec3;
                let dvec: Vec3;
                let rad: f64;
                let gravity: f64;

                let mut lam1: f64 = 0.0;
                let mut lam2: f64 = 0.0;
                let mut ingress: f64 = 0.0;
                let mut egress: f64 = 0.0;

                // Direction is now defined, so calculate radius and thus the
                // position according to whether we are accounting for Roche
                // geometry or not.

                if star == Star::Primary && roche1 {
                    (posn, dvec, rad, gravity) =
                        roche_context1.face(dirn, rref1, pref1, acc).unwrap();
                } else if star == Star::Secondary && roche2 {
                    (posn, dvec, rad, gravity) =
                        roche_context2.face(dirn, rref2, pref2, acc).unwrap();
                } else {
                    // Ignore Roche distortion
                    if star == Star::Primary {
                        rad = r1;
                        posn = cofm1 + rad * dirn;
                    } else {
                        rad = r2;
                        posn = cofm2 + rad * dirn;
                    }
                    dvec = dirn;
                    gravity = 1.0;
                }

                // Area, accounting for angle of face
                let area: f64 = ((g.phi2 - g.phi1) / (g.nphi as f64) * rad * g.sint)
                    * ((thi - tlo) / (g.nl as f64) * rad)
                    / dirn.dot(&dvec);

                // Eclipse computation. We calculate whether a point is
                // eclipsed, and, if it is, its ingress and egress
                // phases. Account for spherical or Roche geometry of other
                // star. Since the stars are convex this calculation only
                // accounts for eclipse by the OTHER star.

                let mut eclipses = Etype::new();

                if eclipse
                    && ((star == Star::Primary
                        && ((roche2
                            && roche_context2
                                .ingress_egress(
                                    ffac2,
                                    iangle,
                                    delta,
                                    &posn,
                                    &mut ingress,
                                    &mut egress,
                                )
                                .unwrap())
                            || (!roche2
                                && roche::sphere_eclipse(
                                    cosi,
                                    sini,
                                    &posn,
                                    &cofm2,
                                    r2,
                                    &mut ingress,
                                    &mut egress,
                                    &mut lam1,
                                    &mut lam2,
                                ))))
                        || (star == Star::Secondary
                            && ((roche1
                                && roche_context1
                                    .ingress_egress(
                                        ffac1,
                                        iangle,
                                        delta,
                                        &posn,
                                        &mut ingress,
                                        &mut egress,
                                    )
                                    .unwrap())
                                || (!roche1
                                    && roche::sphere_eclipse(
                                        cosi,
                                        sini,
                                        &posn,
                                        &cofm1,
                                        r1,
                                        &mut ingress,
                                        &mut egress,
                                        &mut lam1,
                                        &mut lam2,
                                    )))))
                {
                    eclipses.push((ingress, egress));
                }
                band.push(Point::new(posn, dvec, area, gravity / gref, eclipses));
            }

            band
        })
        .collect();


    for band in bands {
        star_grid.extend(band);
    }
}

pub fn star_eclipse(
    roche_context: &RocheContext,
    r: f64,
    ffac: f64,
    iangle: f64,
    posn: &Vec3,
    delta: f64,
    roche: bool,
    star: Star,
    eclipses: &mut Etype,
) -> Result<(), RocheError> {
    let ri: f64 = iangle.to_radians();
    let (sini, cosi) = ri.sin_cos();
    let cofm = match star {
        Star::Primary => Vec3::cofm1(),
        Star::Secondary => Vec3::cofm2(),
    };
    let mut lam1: f64 = 0.0;
    let mut lam2: f64 = 0.0;
    let mut ingress: f64 = 0.0;
    let mut egress: f64 = 0.0;
    // let mut eclipses = Etype::new();
    if (roche
        && roche_context.ingress_egress(ffac, iangle, delta, posn, &mut ingress, &mut egress)?)
        || (!roche
            && roche::sphere_eclipse(
                cosi,
                sini,
                posn,
                &cofm,
                r,
                &mut ingress,
                &mut egress,
                &mut lam1,
                &mut lam2,
            ))
    {
        eclipses.push((ingress, egress));
    }
    Ok(())
}


struct BandGeom {
    sint: f64,
    cost: f64,
    phi1: f64,
    phi2: f64,
    nphi: usize,
    nl: usize,
}

fn band_geometry(
    nt: usize,
    tlo: f64,
    thi: f64,
    dtheta: f64,
    nlat: usize,
    nlat1: usize,
    nlat2: usize,
    infill: bool,
    nlngfill: u32,
    star: Star,
) -> BandGeom {
    if infill {
        if nt < nlat1 {
            let theta = tlo + (thi - tlo) * (nt as f64 + 0.5) / nlat1 as f64;
            let (sint, cost) = theta.sin_cos();

            let nphi = ((PI * sint * (1.0 + nlngfill as f64) / dtheta).max(8.0)) as usize;

            let (phi1, phi2) = if star == Star::Primary {
                (-FRAC_PI_2, FRAC_PI_2)
            } else {
                (FRAC_PI_2, 3.0 * FRAC_PI_2)
            };

            BandGeom {
                sint,
                cost,
                phi1,
                phi2,
                nphi,
                nl: nlat1,
            }
        } else {
            let theta = tlo + (thi - tlo) * ((nt - nlat1) as f64 + 0.5) / nlat2 as f64;
            let (sint, cost) = theta.sin_cos();

            let nphi = ((PI * sint / dtheta).max(8.0)) as usize;

            let (phi1, phi2) = if star == Star::Primary {
                (FRAC_PI_2, 3.0 * FRAC_PI_2)
            } else {
                (-FRAC_PI_2, FRAC_PI_2)
            };

            BandGeom {
                sint,
                cost,
                phi1,
                phi2,
                nphi,
                nl: nlat2,
            }
        }
    } else {
        let theta = tlo + (thi - tlo) * (nt as f64 + 0.5) / nlat as f64;
        let (sint, cost) = theta.sin_cos();

        let nphi = ((2.0 * PI * sint / dtheta).max(16.0)) as usize;

        BandGeom {
            sint,
            cost,
            phi1: 0.0,
            phi2: TAU,
            nphi,
            nl: nlat,
        }
    }
}
