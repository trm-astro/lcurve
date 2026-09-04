use crate::grid::Grid;
use crate::model::Model;
use crate::set_star_grid::star_eclipse;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use roche::errors::RocheError;
use roche::{self, Etype, Point, RocheContext, Star, Vec3};
use std::f64::consts::TAU;
use std::panic;

///
/// set_disc_grid sets up the elements needed to define an accretion
/// disc around the primary star in a co-rotating binary system.  For
/// each element set_disc_grid computes the position within the binary,
/// the direction perpendicular to the element, the area of the element
/// and whether it is eclipsed by the stars. The disc is modelled as
/// having a power law variation of height with radius. Elements are
/// only computed for the top surface of the disc. The area calculation
/// ignores the angle of the elements.
///
/// All numbers assume a separation = 1. The primary star is centred at
/// (0,0,0), the secondary at (1,0,0). The y-axis points in the
/// direction of motion of the secondary star.
///
/// \param model    the Model
/// \param disc      array representing the disc
/// \exception Exceptions are thrown if the specified radii over-fill the
/// Roche lobes.
///
pub fn set_disc_grid(model: &Model) -> Result<Grid, RocheError> {
    const EFAC: f64 = 1.0000001;

    let (mut r1, mut r2) = model.get_r1r2();

    let roche_context1: RocheContext =
        RocheContext::new(model.q.value, Star::Primary, model.spin1.value)?;
    let roche_context2: RocheContext =
        RocheContext::new(model.q.value, Star::Secondary, model.spin2.value)?;

    let rl1: f64 = roche_context1.x_l1;
    if r1 < 0.0 {
        r1 = rl1;
    } else if r1 > rl1 {
        panic!("Primary star is larger than its Roche Lobe!");
    }

    let rl2: f64 = 1.0 - roche_context2.x_l1;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        panic!("Secondary star is larger than its Roche Lobe!");
    }

    // note that the inner radius of the disc is set equal to that of
    // the white dwarf if rdisc1 <= 0 while the outer disc is set
    // equal to the spot radius

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

    // Calculate a reference radius and potential for the two stars
    let ffac1: f64 = r1 / rl1;
    // let (rref1, pref1) = roche_context1.ref_sphere(ffac1);

    let ffac2: f64 = r2 / rl2;
    // let (rref2, pref2) = roche_context2.ref_sphere(ffac2);

    // let cofm1 = Vec3::cofm1();
    // let cofm2 = Vec3::cofm2();

    // let mut ntheta: usize;

    // drrad here introduced to avoid ntheta getting silly if rdisc1 and rdisc2
    // are close to each other.

    let drad: f64 = (rdisc2 - rdisc1) / model.nrad as f64;
    let drrad: f64 = rdisc2 / model.nrad as f64;

    let disc_grid_points: Vec<Point> = (0..model.nrad)
        .into_par_iter()
        .flat_map_iter(|i| {
            let rad: f64 = rdisc1 + (rdisc2 - rdisc1) * (i as f64 + 0.5) / model.nrad as f64;
            let ntheta: usize = ((TAU * rad / drrad).ceil() as usize).max(8);
            let h: f64 = EFAC * model.height_disc.value * rad.powf(model.beta_disc.value);
            let tanp: f64 = model.beta_disc.value * h / rad;
            let cosp: f64 = 1.0 / (1.0 + tanp * tanp).sqrt();
            let sinp: f64 = tanp * cosp;
            // NB: no accounting for the angle of the face
            let area: f64 = TAU / ntheta as f64 * rad * drad;

            (0..ntheta).map(move |j| {
                let theta = TAU * j as f64 / ntheta as f64;
                let (sint, cost) = theta.sin_cos();
                let posn = Vec3::new(rad * cost, rad * sint, h);
                let dirn = Vec3::new(-cost * sinp, -sint * sinp, cosp);

                let mut eclipses = if model.opaque {
                    roche::disc_eclipse(
                        model.iangle.value,
                        rdisc1,
                        rdisc2,
                        model.beta_disc.value,
                        model.height_disc.value,
                        &posn,
                    )
                    .unwrap()
                } else {
                    Etype::new()
                };

                if model.eclipse1 {
                    roche::star_eclipse(
                        model.q.value,
                        model.spin1.value,
                        r1,
                        ffac1,
                        model.iangle.value,
                        &posn,
                        model.delta_phase,
                        model.roche1,
                        Star::Primary,
                        &mut eclipses,
                    )
                    .unwrap()
                }
                if model.eclipse2 {
                    roche::star_eclipse(
                        model.q.value,
                        model.spin2.value,
                        r2,
                        ffac2,
                        model.iangle.value,
                        &posn,
                        model.delta_phase,
                        model.roche2,
                        Star::Secondary,
                        &mut eclipses,
                    )
                    .unwrap()
                }

                Point::new(posn, dirn, area, 1., eclipses)
            })
        })
        .collect();
    Ok(Grid::new(disc_grid_points))
}

///
/// set_disc_edge sets up the elements needed to define a rim of a
/// cyclindrical accretion disc. The elements are assumed to face
/// radially outwards or inwards depending upon whether it is an outer
/// or inner edge, except for elements on the upper edges of the rim
/// which are set to face upwards to guarantee that they will be
/// displayed unless they are eclipsed.
///
/// All numbers assume a separation = 1. The primary star is centred at
/// (0,0,0), the secondary at (1,0,0). The y-axis points in the
/// direction of motion of the secondary star.
///
///
/// \param mdl       input model
/// \param outer     true for the outer edge, false for the inner edge
/// \param edge      array representing the outer edge
/// \param visial    array representing the outer edge
/// \exception Exceptions are thrown if the specified radii over-fill the
/// Roche lobes.
///
pub fn set_disc_edge_grid(
    model: &Model,
    outer: bool,
    visual: bool,
) -> Result<Grid, RocheError> {
    const EFAC: f64 = 1.0000001;

    let (mut r1, mut r2) = model.get_r1r2();

    let roche_context1 = RocheContext::new(model.q.value, Star::Primary, model.spin1.value)?;
    let roche_context2 = RocheContext::new(model.q.value, Star::Secondary, model.spin2.value)?;

    let rl1: f64 = roche_context1.x_l1;
    if r1 < 0.0 {
        r1 = rl1;
    } else if r1 > rl1 {
        panic!("Primary star is larger than its Roche Lobe!");
    }

    let rl2: f64 = 1.0 - roche_context2.x_l1;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        panic!("Secondary star is larger than its Roche Lobe!");
    }

    // note that the inner radius of the disc is set equal to that of
    // the white dwarf if rdisc1 <= 0 while the outer disc is set
    // equal to the spot radius

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
    let size: f64 = rdisc2 / model.nrad as f64;

    // // Calculate a reference radius and potential for the two stars
    let ffac1: f64 = r1 / rl1;
    // let (rref1, pref1) = roche_context1.ref_sphere(ffac1);

    let ffac2: f64 = r2 / rl2;
    // let (rref2, pref2) = roche_context2.ref_sphere(ffac2);

    // let cofm1: Vec3 = Vec3::cofm1();
    // let cofm2: Vec3 = Vec3::cofm2();

    let rad: f64 = if outer { rdisc2 } else { rdisc1 };
    let h: f64 = model.height_disc.value * rad.powf(model.beta_disc.value);
    let nout = (2.0 * h / size) as usize + 2;
    let mut ntheta: usize = (TAU * rad / size).ceil() as usize;
    ntheta = if ntheta > 8 { ntheta } else { 8 };

    let mut edge_grid: Vec<Point> = Vec::with_capacity(ntheta * nout);
    edge_grid.resize(ntheta * nout, Point::default());

    let mut dirn: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    let mut posn: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    let area: f64 = TAU / ntheta as f64 * rad * 2.0 * h / (nout - 1) as f64;

    let mut theta: f64;
    let mut sint: f64;
    let mut cost: f64;

    let mut eclipses: Etype = Etype::new();
    let mut nface: usize = 0;
    for i in 0..ntheta {
        theta = TAU * i as f64 / ntheta as f64;
        (sint, cost) = theta.sin_cos();

        // upper rim element
        if outer {
            posn.set(EFAC * rad * cost, EFAC * rad * sint, EFAC * h);
        } else {
            posn.set(rad * cost / EFAC, rad * sint / EFAC, EFAC * h);
        }

        // Direction set so that it will always be visible if is not
        // eclipsed if for visualisation

        if visual {
            dirn.set(0.0, 0.0, 0.0);
        } else {
            dirn.set(cost, sint, 0.0);
        }

        if model.opaque {
            eclipses = roche::disc_eclipse(
                model.iangle.value,
                rdisc1,
                rdisc2,
                model.beta_disc.value,
                model.height_disc.value,
                &posn,
            )?;
        } else {
            eclipses.clear();
        }

        // Primary star can expand to wipe out inner disc; trap but report such errors
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
            // might need to add the error catching here
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
        edge_grid[nface] = Point::new(posn, dirn, area / 2.0, 1.0, eclipses.clone());
        nface += 1;

        // All lower elements
        for j in 0..nout - 1 {
            if outer {
                posn.set(
                    EFAC * rad * cost,
                    EFAC * rad * sint,
                    -h + 2.0 * h * j as f64 / (nout as f64 - 1.0),
                );
                dirn.set(cost, sint, 0.0);
            } else {
                posn.set(
                    rad * cost / EFAC,
                    rad * sint / EFAC,
                    -h + 2.0 * h * j as f64 / (nout as f64 - 1.0),
                );
                dirn.set(-cost, -sint, 0.0);
            }

            if model.opaque {
                eclipses = roche::disc_eclipse(
                    model.iangle.value,
                    rdisc1,
                    rdisc2,
                    model.beta_disc.value,
                    model.height_disc.value,
                    &posn,
                )?;
            } else {
                eclipses.clear();
            }

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
            edge_grid[nface] = Point::new(posn, dirn, area, 1.0, eclipses.clone());
            nface += 1;
        }
    }

    Ok(Grid::new(edge_grid))
}
