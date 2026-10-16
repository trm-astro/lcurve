use pyo3::prelude::*;
use pyo3::types::PyAny;
use roche::{constants::C, x_l1_1, x_l1_2, errors::RocheError};
use serde::{Deserialize, Serialize};
use serde_pyobject::from_pyobject;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::{File, write};
use std::io::{self, BufRead};
use std::path::Path;
use crate::ldc::{LDC, LDCType};
use crate::pparam::{Pparam, PparamPartial};

macro_rules! apply_update {
    ($self:ident, $upd:ident, {
        $(
            $field:ident : $kind:ident
        ),* $(,)?
    }) => {
        $(
            apply_update!(@field $self, $upd, $field, $kind);
        )*
    };

    // --- plain types (f64, bool, etc.) ---
    (@field $self:ident, $upd:ident, $field:ident, plain) => {
        if let Some(v) = $upd.$field {
            $self.$field = v;
        }
    };

    // --- Pparam ---
    (@field $self:ident, $upd:ident, $field:ident, pparam) => {
        if let Some(v) = $upd.$field {
            match v {
                PparamUpdate::Full(p) => $self.$field = p,
                PparamUpdate::Partial(p) => {
                    if let Some(v) = p.value { $self.$field.value = v; }
                    if let Some(v) = p.range { $self.$field.range = v; }
                    if let Some(v) = p.dstep { $self.$field.dstep = v; }
                    if let Some(v) = p.vary { $self.$field.vary = v; }
                    if let Some(v) = p.defined { $self.$field.defined = v; }
                }
                PparamUpdate::Value(val) => {
                    $self.$field.value = val;
                    $self.$field.defined = true;
                },
            }
        }
    };

    // enum
    (@field $self:ident, $upd:ident, $field:ident, enum) => {
        if let Some(v) = $upd.$field {
            $self.$field = v;
        }
    };
}

#[derive(Debug)]
pub enum Entry {
    Param(Pparam),
    Scalar(String),
}

#[derive(Clone, Copy, Deserialize)]
#[serde(untagged)]
pub enum PparamUpdate {
    Full(Pparam),
    Partial(PparamPartial),
    Value(f64),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelUpdate {
    pub q: Option<PparamUpdate>,
    pub iangle: Option<PparamUpdate>,
    pub r1: Option<PparamUpdate>,
    pub r2: Option<PparamUpdate>,
    pub cphi3: Option<PparamUpdate>,
    pub cphi4: Option<PparamUpdate>,
    pub spin1: Option<PparamUpdate>,
    pub spin2: Option<PparamUpdate>,
    pub t1: Option<PparamUpdate>,
    pub t2: Option<PparamUpdate>,
    pub ldc1_1: Option<PparamUpdate>,
    pub ldc1_2: Option<PparamUpdate>,
    pub ldc1_3: Option<PparamUpdate>,
    pub ldc1_4: Option<PparamUpdate>,
    pub ldc2_1: Option<PparamUpdate>,
    pub ldc2_2: Option<PparamUpdate>,
    pub ldc2_3: Option<PparamUpdate>,
    pub ldc2_4: Option<PparamUpdate>,
    pub velocity_scale: Option<PparamUpdate>,
    pub beam_factor1: Option<PparamUpdate>,
    pub beam_factor2: Option<PparamUpdate>,
    pub t0: Option<PparamUpdate>,
    pub period: Option<PparamUpdate>,
    pub pdot: Option<PparamUpdate>,
    pub deltat: Option<PparamUpdate>,
    pub gravity_dark1: Option<PparamUpdate>,
    pub gravity_dark2: Option<PparamUpdate>,
    pub absorb: Option<PparamUpdate>,
    pub slope: Option<PparamUpdate>,
    pub quad: Option<PparamUpdate>,
    pub cube: Option<PparamUpdate>,
    pub third: Option<PparamUpdate>,
    pub rdisc1: Option<PparamUpdate>,
    pub rdisc2: Option<PparamUpdate>,
    pub height_disc: Option<PparamUpdate>,
    pub beta_disc: Option<PparamUpdate>,
    pub temp_disc: Option<PparamUpdate>,
    pub texp_disc: Option<PparamUpdate>,
    pub lin_limb_disc: Option<PparamUpdate>,
    pub quad_limb_disc: Option<PparamUpdate>,
    pub temp_edge: Option<PparamUpdate>,
    pub absorb_edge: Option<PparamUpdate>,
    pub radius_spot: Option<PparamUpdate>,
    pub length_spot: Option<PparamUpdate>,
    pub height_spot: Option<PparamUpdate>,
    pub expon_spot: Option<PparamUpdate>,
    pub epow_spot: Option<PparamUpdate>,
    pub angle_spot: Option<PparamUpdate>,
    pub yaw_spot: Option<PparamUpdate>,
    pub temp_spot: Option<PparamUpdate>,
    pub tilt_spot: Option<PparamUpdate>,
    pub cfrac_spot: Option<PparamUpdate>,
    pub stsp11_long: Option<PparamUpdate>,
    pub stsp11_lat: Option<PparamUpdate>,
    pub stsp11_fwhm: Option<PparamUpdate>,
    pub stsp11_tcen: Option<PparamUpdate>,
    pub stsp12_long: Option<PparamUpdate>,
    pub stsp12_lat: Option<PparamUpdate>,
    pub stsp12_fwhm: Option<PparamUpdate>,
    pub stsp12_tcen: Option<PparamUpdate>,
    pub stsp13_long: Option<PparamUpdate>,
    pub stsp13_lat: Option<PparamUpdate>,
    pub stsp13_fwhm: Option<PparamUpdate>,
    pub stsp13_tcen: Option<PparamUpdate>,
    pub stsp21_long: Option<PparamUpdate>,
    pub stsp21_lat: Option<PparamUpdate>,
    pub stsp21_fwhm: Option<PparamUpdate>,
    pub stsp21_tcen: Option<PparamUpdate>,
    pub stsp22_long: Option<PparamUpdate>,
    pub stsp22_lat: Option<PparamUpdate>,
    pub stsp22_fwhm: Option<PparamUpdate>,
    pub stsp22_tcen: Option<PparamUpdate>,
    pub uesp_long1: Option<PparamUpdate>,
    pub uesp_long2: Option<PparamUpdate>,
    pub uesp_lathw: Option<PparamUpdate>,
    pub uesp_taper: Option<PparamUpdate>,
    pub uesp_temp: Option<PparamUpdate>,
    pub delta_phase: Option<f64>,
    pub nlat1f: Option<u32>,
    pub nlat2f: Option<u32>,
    pub nlat1c: Option<u32>,
    pub nlat2c: Option<u32>,
    pub npole: Option<bool>,
    pub nlatfill: Option<u32>,
    pub nlngfill: Option<u32>,
    pub lfudge: Option<f64>,
    pub llo: Option<f64>,
    pub lhi: Option<f64>,
    pub phase1: Option<f64>,
    pub phase2: Option<f64>,
    pub wavelength: Option<f64>,
    pub roche1: Option<bool>,
    pub roche2: Option<bool>,
    pub eclipse1: Option<bool>,
    pub eclipse2: Option<bool>,
    pub glens1: Option<bool>,
    pub use_radii: Option<bool>,
    pub tperiod: Option<f64>,
    pub gdark_bolom1: Option<bool>,
    pub gdark_bolom2: Option<bool>,
    pub mucrit1: Option<f64>,
    pub mucrit2: Option<f64>,
    pub limb1: Option<LDCType>,
    pub limb2: Option<LDCType>,
    pub mirror: Option<bool>,
    pub add_disc: Option<bool>,
    pub nrad: Option<u32>,
    pub opaque: Option<bool>,
    pub add_spot: Option<bool>,
    pub nspot: Option<u32>,
    pub iscale: Option<bool>,
}

impl ModelUpdate {

    pub fn check_bounds(&self) -> Result<(), RocheError> {
        const MAX_TEMP: f64 = 1.0e10;

        if let Some(iangle) = self.iangle {
            check_pparam_update_or_error(iangle, 0.0, 90.0, "iangle must be between 0.0 and 90.0.")?;
        }

        if let Some(q) = self.q {
            check_pparam_update_or_error(q, f64::MIN_POSITIVE, f64::INFINITY, "q must be positive and non-zero.")?;
        }

        if let Some(r1) = self.r1 {
            check_pparam_update_or_error(r1, f64::MIN_POSITIVE, 1.0, "r1 must be non-zero and between 0.0 and 1.0.")?;
        }

        if let Some(r2) = self.r2 {
            if !check_pparam_update(r2, f64::MIN_POSITIVE, 1.0) && !check_pparam_update(r2, -1.0, -1.0) {
                return Err(RocheError::ParameterError("r2 must be between 0.0 and 1.0 or set to -1.0 if Roche-filling.".to_string()));
            }
        }

        if let Some(cphi3) = self.cphi3 {
            check_pparam_update_or_error(cphi3, 0.0, 0.25, "cphi3 must be between 0.0 and 0.25.")?;
        }

        if let Some(cphi4) = self.cphi4 {
            check_pparam_update_or_error(cphi4, 0.0, 0.25, "cphi4 must be between 0.0 and 0.25.")?;
        }

        if let Some(t1) = self.t1 {
            check_pparam_update_or_error(t1, 0.0, MAX_TEMP, "t1 must be between 0.0 and 1.0e10.")?;
        }

        if let Some(t2) = self.t2 {
            check_pparam_update_or_error(t2, 0.0, MAX_TEMP, "t2 must be between 0.0 and 1.0e10.")?;
        }

        if let Some(velocity_scale) = self.velocity_scale {
            check_pparam_update_or_error(velocity_scale, f64::MIN_POSITIVE, C/1.0e3, "velocity_scale must be between 0.0 and the speed of light (km/s).")?;
        }

        if let Some(period) = self.period {
            check_pparam_update_or_error(period, f64::MIN_POSITIVE, f64::INFINITY, "period must be positive and non-zero.")?;
        }

        if let Some(absorb) = self.absorb {
            check_pparam_update_or_error(absorb, 0.0, 1.0, "absorb must be between 0.0 and 1.0.")?;
        }

        if let Some(third) = self.third {
            check_pparam_update_or_error(third, 0.0, f64::INFINITY, "third must be positive.")?;
        }

        if let Some(rdisc1) = self.rdisc1 {
            if !check_pparam_update(rdisc1, 0.0, 1.0) && !check_pparam_update(rdisc1, -1.0, -1.0) {
                return Err(RocheError::ParameterError("rdisc1 must be between 0.0 and 1.0 or set to -1.0 if locked to radius of star 1.".to_string()));
            }
        }

        if let Some(rdisc2) = self.rdisc2 {
            if !check_pparam_update(rdisc2, 0.0, 1.0) && !check_pparam_update(rdisc2, -1.0, -1.0) {
                return Err(RocheError::ParameterError("rdisc2 must be between 0.0 and 1.0 or set to -1.0 if locked to bright spot radius.".to_string()));
            }
        }

        if let Some(temp_disc) = self.temp_disc {
            check_pparam_update_or_error(temp_disc, 0.0, MAX_TEMP, "temp_disc must be between 0.0 and 1.0e10.")?;
        }

        if let Some(temp_edge) = self.temp_edge {
            check_pparam_update_or_error(temp_edge, 0.0, MAX_TEMP, "temp_edge must be between 0.0 and 1.0e10.")?;
        }

        if let Some(absorb_edge) = self.absorb_edge {
            check_pparam_update_or_error(absorb_edge, 0.0, 1.0, "absorb_edge must be between 0.0 and 1.0.")?;
        }

        if let Some(radius_spot) = self.radius_spot {
            check_pparam_update_or_error(radius_spot, f64::MIN_POSITIVE, 1.0, "radius_spot must be positive, non-zero, and 1.0.")?;
        }

        if let Some(temp_spot) = self.temp_spot {
            check_pparam_update_or_error(temp_spot, 0.0, MAX_TEMP, "temp_spot must be between 0.0 and 1.0e10.")?;
        }

        if let Some(tperiod) = self.tperiod {
            check_parameter_f64_or_error(tperiod, f64::MIN_POSITIVE, f64::INFINITY, "tperiod must be positive and non-zero")?;
        }
        
        if let Some(wavelength) = self.wavelength {
            check_parameter_f64_or_error(wavelength, f64::MIN_POSITIVE, f64::INFINITY, "wavelength must be positive and non-zero")?;
        }
        
        Ok(())
    }


    pub fn grid_changed(&self) -> bool {
        self.q.is_some()
            || self.iangle.is_some()
            || self.r1.is_some()
            || self.r2.is_some()
            || self.cphi3.is_some()
            || self.cphi4.is_some()
            || self.spin1.is_some()
            || self.spin2.is_some()
            || self.t0.is_some()
            || self.period.is_some()
            || self.pdot.is_some()
            || self.deltat.is_some()
            || self.rdisc1.is_some()
            || self.rdisc2.is_some()
            || self.radius_spot.is_some()
            || self.height_spot.is_some()
            || self.expon_spot.is_some()
            || self.epow_spot.is_some()
            || self.angle_spot.is_some()
            || self.yaw_spot.is_some()
            || self.temp_spot.is_some()
            || self.tilt_spot.is_some()
            || self.cfrac_spot.is_some()
            || self.delta_phase.is_some()
            || self.nlat1f.is_some()
            || self.nlat2f.is_some()
            || self.nlat1c.is_some()
            || self.nlat2c.is_some()
            || self.npole.is_some()
            || self.nlatfill.is_some()
            || self.nlngfill.is_some()
            || self.lfudge.is_some()
            || self.llo.is_some()
            || self.lhi.is_some()
            || self.phase1.is_some()
            || self.phase2.is_some()
            || self.roche1.is_some()
            || self.roche2.is_some()
            || self.eclipse1.is_some()
            || self.eclipse2.is_some()
            || self.use_radii.is_some()
            || self.add_disc.is_some()
            || self.nrad.is_some()
            || self.opaque.is_some()
            || self.add_spot.is_some()
            || self.nspot.is_some()
    }
}

#[pyclass(from_py_object, module = "lcurve")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct Model {
    #[pyo3(get)]
    pub q: Pparam,
    #[pyo3(get)]
    pub iangle: Pparam,
    #[pyo3(get)]
    pub r1: Pparam,
    #[pyo3(get)]
    pub r2: Pparam,
    #[pyo3(get)]
    pub cphi3: Pparam,
    #[pyo3(get)]
    pub cphi4: Pparam,
    #[pyo3(get)]
    pub spin1: Pparam,
    #[pyo3(get)]
    pub spin2: Pparam,
    #[pyo3(get)]
    pub t1: Pparam,
    #[pyo3(get)]
    pub t2: Pparam,
    #[pyo3(get)]
    pub ldc1_1: Pparam,
    #[pyo3(get)]
    pub ldc1_2: Pparam,
    #[pyo3(get)]
    pub ldc1_3: Pparam,
    #[pyo3(get)]
    pub ldc1_4: Pparam,
    #[pyo3(get)]
    pub ldc2_1: Pparam,
    #[pyo3(get)]
    pub ldc2_2: Pparam,
    #[pyo3(get)]
    pub ldc2_3: Pparam,
    #[pyo3(get)]
    pub ldc2_4: Pparam,
    #[pyo3(get)]
    pub velocity_scale: Pparam,
    #[pyo3(get)]
    pub beam_factor1: Pparam,
    #[pyo3(get)]
    pub beam_factor2: Pparam,
    #[pyo3(get)]
    pub t0: Pparam,
    #[pyo3(get)]
    pub period: Pparam,
    #[pyo3(get)]
    pub pdot: Pparam,
    #[pyo3(get)]
    pub deltat: Pparam,
    #[pyo3(get)]
    pub gravity_dark1: Pparam,
    #[pyo3(get)]
    pub gravity_dark2: Pparam,
    #[pyo3(get)]
    pub absorb: Pparam,
    #[pyo3(get)]
    pub slope: Pparam,
    #[pyo3(get)]
    pub quad: Pparam,
    #[pyo3(get)]
    pub cube: Pparam,
    #[pyo3(get)]
    pub third: Pparam,
    #[pyo3(get)]
    pub rdisc1: Pparam,
    #[pyo3(get)]
    pub rdisc2: Pparam,
    #[pyo3(get)]
    pub height_disc: Pparam,
    #[pyo3(get)]
    pub beta_disc: Pparam,
    #[pyo3(get)]
    pub temp_disc: Pparam,
    #[pyo3(get)]
    pub texp_disc: Pparam,
    #[pyo3(get)]
    pub lin_limb_disc: Pparam,
    #[pyo3(get)]
    pub quad_limb_disc: Pparam,
    #[pyo3(get)]
    pub temp_edge: Pparam,
    #[pyo3(get)]
    pub absorb_edge: Pparam,
    #[pyo3(get)]
    pub radius_spot: Pparam,
    #[pyo3(get)]
    pub length_spot: Pparam,
    #[pyo3(get)]
    pub height_spot: Pparam,
    #[pyo3(get)]
    pub expon_spot: Pparam,
    #[pyo3(get)]
    pub epow_spot: Pparam,
    #[pyo3(get)]
    pub angle_spot: Pparam,
    #[pyo3(get)]
    pub yaw_spot: Pparam,
    #[pyo3(get)]
    pub temp_spot: Pparam,
    #[pyo3(get)]
    pub tilt_spot: Pparam,
    #[pyo3(get)]
    pub cfrac_spot: Pparam,
    #[pyo3(get)]
    pub stsp11_long: Pparam,
    #[pyo3(get)]
    pub stsp11_lat: Pparam,
    #[pyo3(get)]
    pub stsp11_fwhm: Pparam,
    #[pyo3(get)]
    pub stsp11_tcen: Pparam,
    #[pyo3(get)]
    pub stsp12_long: Pparam,
    #[pyo3(get)]
    pub stsp12_lat: Pparam,
    #[pyo3(get)]
    pub stsp12_fwhm: Pparam,
    #[pyo3(get)]
    pub stsp12_tcen: Pparam,
    #[pyo3(get)]
    pub stsp13_long: Pparam,
    #[pyo3(get)]
    pub stsp13_lat: Pparam,
    #[pyo3(get)]
    pub stsp13_fwhm: Pparam,
    #[pyo3(get)]
    pub stsp13_tcen: Pparam,
    #[pyo3(get)]
    pub stsp21_long: Pparam,
    #[pyo3(get)]
    pub stsp21_lat: Pparam,
    #[pyo3(get)]
    pub stsp21_fwhm: Pparam,
    #[pyo3(get)]
    pub stsp21_tcen: Pparam,
    #[pyo3(get)]
    pub stsp22_long: Pparam,
    #[pyo3(get)]
    pub stsp22_lat: Pparam,
    #[pyo3(get)]
    pub stsp22_fwhm: Pparam,
    #[pyo3(get)]
    pub stsp22_tcen: Pparam,
    #[pyo3(get)]
    pub uesp_long1: Pparam,
    #[pyo3(get)]
    pub uesp_long2: Pparam,
    #[pyo3(get)]
    pub uesp_lathw: Pparam,
    #[pyo3(get)]
    pub uesp_taper: Pparam,
    #[pyo3(get)]
    pub uesp_temp: Pparam,
    #[pyo3(get)]
    pub delta_phase: f64,
    #[pyo3(get)]
    pub nlat1f: u32,
    #[pyo3(get)]
    pub nlat2f: u32,
    #[pyo3(get)]
    pub nlat1c: u32,
    #[pyo3(get)]
    pub nlat2c: u32,
    #[pyo3(get)]
    pub npole: bool,
    #[pyo3(get)]
    pub nlatfill: u32,
    #[pyo3(get)]
    pub nlngfill: u32,
    #[pyo3(get)]
    pub lfudge: f64,
    #[pyo3(get)]
    pub llo: f64,
    #[pyo3(get)]
    pub lhi: f64,
    #[pyo3(get)]
    pub phase1: f64,
    #[pyo3(get)]
    pub phase2: f64,
    #[pyo3(get)]
    pub wavelength: f64,
    #[pyo3(get)]
    pub roche1: bool,
    #[pyo3(get)]
    pub roche2: bool,
    #[pyo3(get)]
    pub eclipse1: bool,
    #[pyo3(get)]
    pub eclipse2: bool,
    #[pyo3(get)]
    pub glens1: bool,
    #[pyo3(get)]
    pub use_radii: bool,
    #[pyo3(get)]
    pub tperiod: f64,
    #[pyo3(get)]
    pub gdark_bolom1: bool,
    #[pyo3(get)]
    pub gdark_bolom2: bool,
    #[pyo3(get)]
    pub mucrit1: f64,
    #[pyo3(get)]
    pub mucrit2: f64,
    #[pyo3(get)]
    pub limb1: LDCType,
    #[pyo3(get)]
    pub limb2: LDCType,
    #[pyo3(get)]
    pub mirror: bool,
    #[pyo3(get)]
    pub add_disc: bool,
    #[pyo3(get)]
    pub nrad: u32,
    #[pyo3(get)]
    pub opaque: bool,
    #[pyo3(get)]
    pub add_spot: bool,
    #[pyo3(get)]
    pub nspot: u32,
    #[pyo3(get)]
    pub iscale: bool,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            q: default_pparam(),
            iangle: default_pparam(),
            r1: default_pparam(),
            r2: default_pparam(),
            cphi3: default_pparam(),
            cphi4: default_pparam(),
            spin1: default_pparam(),
            spin2: default_pparam(),
            t1: default_pparam(),
            t2: default_pparam(),
            ldc1_1: default_pparam(),
            ldc1_2: default_pparam(),
            ldc1_3: default_pparam(),
            ldc1_4: default_pparam(),
            ldc2_1: default_pparam(),
            ldc2_2: default_pparam(),
            ldc2_3: default_pparam(),
            ldc2_4: default_pparam(),
            velocity_scale: default_pparam(),
            beam_factor1: default_pparam(),
            beam_factor2: default_pparam(),
            t0: default_pparam(),
            period: default_pparam(),
            pdot: default_pparam(),
            deltat: default_pparam(),
            gravity_dark1: default_pparam(),
            gravity_dark2: default_pparam(),
            absorb: default_pparam(),
            slope: default_pparam(),
            quad: default_pparam(),
            cube: default_pparam(),
            third: default_pparam(),
            rdisc1: default_pparam(),
            rdisc2: default_pparam(),
            height_disc: default_pparam(),
            beta_disc: default_pparam(),
            temp_disc: default_pparam(),
            texp_disc: default_pparam(),
            lin_limb_disc: default_pparam(),
            quad_limb_disc: default_pparam(),
            temp_edge: default_pparam(),
            absorb_edge: default_pparam(),
            radius_spot: default_pparam(),
            length_spot: default_pparam(),
            height_spot: default_pparam(),
            expon_spot: default_pparam(),
            epow_spot: default_pparam(),
            angle_spot: default_pparam(),
            yaw_spot: default_pparam(),
            temp_spot: default_pparam(),
            tilt_spot: default_pparam(),
            cfrac_spot: default_pparam(),
            stsp11_long: default_pparam(),
            stsp11_lat: default_pparam(),
            stsp11_fwhm: default_pparam(),
            stsp11_tcen: default_pparam(),
            stsp12_long: default_pparam(),
            stsp12_lat: default_pparam(),
            stsp12_fwhm: default_pparam(),
            stsp12_tcen: default_pparam(),
            stsp13_long: default_pparam(),
            stsp13_lat: default_pparam(),
            stsp13_fwhm: default_pparam(),
            stsp13_tcen: default_pparam(),
            stsp21_long: default_pparam(),
            stsp21_lat: default_pparam(),
            stsp21_fwhm: default_pparam(),
            stsp21_tcen: default_pparam(),
            stsp22_long: default_pparam(),
            stsp22_lat: default_pparam(),
            stsp22_fwhm: default_pparam(),
            stsp22_tcen: default_pparam(),
            uesp_long1: default_pparam(),
            uesp_long2: default_pparam(),
            uesp_lathw: default_pparam(),
            uesp_taper: default_pparam(),
            uesp_temp: default_pparam(),
            delta_phase: default_delta_phase(),
            nlat1f: default_ten(),
            nlat2f: default_ten(),
            nlat1c: default_ten(),
            nlat2c: default_ten(),
            npole: default_false(),
            nlatfill: default_zero(),
            nlngfill: default_ten(),
            lfudge: default_zero_f64(),
            llo: default_zero_f64(),
            lhi: default_zero_f64(),
            phase1: default_zero_f64(),
            phase2: default_zero_f64(),
            wavelength: default_zero_f64(),
            roche1: default_false(),
            roche2: default_false(),
            eclipse1: default_true(),
            eclipse2: default_true(),
            glens1: default_false(),
            use_radii: default_true(),
            tperiod: default_zero_f64(),
            gdark_bolom1: default_false(),
            gdark_bolom2: default_false(),
            mucrit1: default_zero_f64(),
            mucrit2: default_zero_f64(),
            limb1: default_poly(),
            limb2: default_poly(),
            mirror: default_false(),
            add_disc: default_false(),
            nrad: default_zero(),
            opaque: default_false(),
            add_spot: default_false(),
            nspot: default_zero(),
            iscale: default_false(),
        }
    }
}

impl Model {
    pub fn from_map(map: HashMap<String, Entry>) -> Result<Self, String> {
        Ok(Self {
            // Pparams
            q: get_p(&map, "q")?,
            iangle: get_p(&map, "iangle")?,
            r1: get_p(&map, "r1")?,
            r2: get_p(&map, "r2")?,
            cphi3: get_p(&map, "cphi3")?,
            cphi4: get_p(&map, "cphi4")?,
            spin1: get_p(&map, "spin1")?,
            spin2: get_p(&map, "spin2")?,
            t1: get_p(&map, "t1")?,
            t2: get_p(&map, "t2")?,
            ldc1_1: get_p(&map, "ldc1_1")?,
            ldc1_2: get_p(&map, "ldc1_2")?,
            ldc1_3: get_p(&map, "ldc1_3")?,
            ldc1_4: get_p(&map, "ldc1_4")?,
            ldc2_1: get_p(&map, "ldc2_1")?,
            ldc2_2: get_p(&map, "ldc2_2")?,
            ldc2_3: get_p(&map, "ldc2_3")?,
            ldc2_4: get_p(&map, "ldc2_4")?,
            velocity_scale: get_p(&map, "velocity_scale")?,
            beam_factor1: get_p(&map, "beam_factor1")?,
            beam_factor2: get_p(&map, "beam_factor2")?,
            t0: get_p(&map, "t0")?,
            period: get_p(&map, "period")?,
            pdot: get_p(&map, "pdot")?,
            deltat: get_p(&map, "deltat")?,
            gravity_dark1: get_p(&map, "gravity_dark1")?,
            gravity_dark2: get_p(&map, "gravity_dark2")?,
            absorb: get_p(&map, "absorb")?,
            slope: get_p(&map, "slope")?,
            quad: get_p(&map, "quad")?,
            cube: get_p(&map, "cube")?,
            third: get_p(&map, "third")?,
            rdisc1: get_p(&map, "rdisc1")?,
            rdisc2: get_p(&map, "rdisc2")?,
            height_disc: get_p(&map, "height_disc")?,
            beta_disc: get_p(&map, "beta_disc")?,
            temp_disc: get_p(&map, "temp_disc")?,
            texp_disc: get_p(&map, "texp_disc")?,
            lin_limb_disc: get_p(&map, "lin_limb_disc")?,
            quad_limb_disc: get_p(&map, "quad_limb_disc")?,
            temp_edge: get_p(&map, "temp_edge").unwrap_or_default(),
            absorb_edge: get_p(&map, "absorb_edge").unwrap_or_default(),
            radius_spot: get_p(&map, "radius_spot")?,
            length_spot: get_p(&map, "length_spot")?,
            height_spot: get_p(&map, "height_spot")?,
            expon_spot: get_p(&map, "expon_spot")?,
            epow_spot: get_p(&map, "epow_spot")?,
            angle_spot: get_p(&map, "angle_spot")?,
            yaw_spot: get_p(&map, "yaw_spot")?,
            temp_spot: get_p(&map, "temp_spot")?,
            tilt_spot: get_p(&map, "tilt_spot")?,
            cfrac_spot: get_p(&map, "cfrac_spot")?,

            stsp11_long: get_p(&map, "stsp11_long").unwrap_or_default(),
            stsp11_lat: get_p(&map, "stsp11_lat").unwrap_or_default(),
            stsp11_fwhm: get_p(&map, "stsp11_fwhm").unwrap_or_default(),
            stsp11_tcen: get_p(&map, "stsp11_tcen").unwrap_or_default(),

            stsp12_long: get_p(&map, "stsp12_long").unwrap_or_default(),
            stsp12_lat: get_p(&map, "stsp12_lat").unwrap_or_default(),
            stsp12_fwhm: get_p(&map, "stsp12_fwhm").unwrap_or_default(),
            stsp12_tcen: get_p(&map, "stsp12_tcen").unwrap_or_default(),

            stsp13_long: get_p(&map, "stsp13_long").unwrap_or_default(),
            stsp13_lat: get_p(&map, "stsp13_lat").unwrap_or_default(),
            stsp13_fwhm: get_p(&map, "stsp13_fwhm").unwrap_or_default(),
            stsp13_tcen: get_p(&map, "stsp13_tcen").unwrap_or_default(),

            stsp21_long: get_p(&map, "stsp21_long").unwrap_or_default(),
            stsp21_lat: get_p(&map, "stsp21_lat").unwrap_or_default(),
            stsp21_fwhm: get_p(&map, "stsp21_fwhm").unwrap_or_default(),
            stsp21_tcen: get_p(&map, "stsp21_tcen").unwrap_or_default(),

            stsp22_long: get_p(&map, "stsp22_long").unwrap_or_default(),
            stsp22_lat: get_p(&map, "stsp22_lat").unwrap_or_default(),
            stsp22_fwhm: get_p(&map, "stsp22_fwhm").unwrap_or_default(),
            stsp22_tcen: get_p(&map, "stsp22_tcen").unwrap_or_default(),

            uesp_long1: get_p(&map, "uesp_long1").unwrap_or_default(),
            uesp_long2: get_p(&map, "uesp_long2").unwrap_or_default(),
            uesp_lathw: get_p(&map, "uesp_lathw").unwrap_or_default(),
            uesp_taper: get_p(&map, "uesp_taper").unwrap_or_default(),
            uesp_temp: get_p(&map, "uesp_temp").unwrap_or_default(),

            // Scalars
            delta_phase: get_f64(&map, "delta_phase")?,
            nlat1f: get_u32(&map, "nlat1f")?,
            nlat2f: get_u32(&map, "nlat2f")?,
            nlat1c: get_u32(&map, "nlat1c")?,
            nlat2c: get_u32(&map, "nlat2c")?,
            npole: get_bool(&map, "npole")?,
            nlatfill: get_u32(&map, "nlatfill")?,
            nlngfill: get_u32(&map, "nlngfill")?,
            lfudge: get_f64(&map, "lfudge")?,
            llo: get_f64(&map, "llo")?,
            lhi: get_f64(&map, "lhi")?,
            phase1: get_f64(&map, "phase1")?,
            phase2: get_f64(&map, "phase2")?,
            wavelength: get_f64(&map, "wavelength")?,
            roche1: get_bool(&map, "roche1")?,
            roche2: get_bool(&map, "roche2")?,
            eclipse1: get_bool(&map, "eclipse1")?,
            eclipse2: get_bool(&map, "eclipse2")?,
            glens1: get_bool(&map, "glens1")?,
            use_radii: get_bool(&map, "use_radii")?,
            tperiod: get_f64(&map, "tperiod")?,
            gdark_bolom1: get_bool(&map, "gdark_bolom1")?,
            gdark_bolom2: get_bool(&map, "gdark_bolom2")?,
            mucrit1: get_f64(&map, "mucrit1")?,
            mucrit2: get_f64(&map, "mucrit2")?,
            limb1: get_ldc(&map, "limb1")?,
            limb2: get_ldc(&map, "limb2")?,
            mirror: get_bool(&map, "mirror")?,
            add_disc: get_bool(&map, "add_disc")?,
            nrad: get_u32(&map, "nrad")?,
            opaque: get_bool(&map, "opaque")?,
            add_spot: get_bool(&map, "add_spot")?,
            nspot: get_u32(&map, "nspot")?,
            iscale: get_bool(&map, "iscale")?,
        })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let map = load_entries(path)?;
        Self::from_map(map)
    }

    pub fn write(&self, path: &str) -> Result<(), String> {
        let mut out = String::new();

        write_param_line(&mut out, "q", &self.q);
        write_param_line(&mut out, "iangle", &self.iangle);
        write_param_line(&mut out, "r1", &self.r1);
        write_param_line(&mut out, "r2", &self.r2);
        write_param_line(&mut out, "cphi3", &self.cphi3);
        write_param_line(&mut out, "cphi4", &self.cphi4);
        write_param_line(&mut out, "spin1", &self.spin1);
        write_param_line(&mut out, "spin2", &self.spin2);
        write_param_line(&mut out, "t1", &self.t1);
        write_param_line(&mut out, "t2", &self.t2);
        write_param_line(&mut out, "ldc1_1", &self.ldc1_1);
        write_param_line(&mut out, "ldc1_2", &self.ldc1_2);
        write_param_line(&mut out, "ldc1_3", &self.ldc1_3);
        write_param_line(&mut out, "ldc1_4", &self.ldc1_4);
        write_param_line(&mut out, "ldc2_1", &self.ldc2_1);
        write_param_line(&mut out, "ldc2_2", &self.ldc2_2);
        write_param_line(&mut out, "ldc2_3", &self.ldc2_3);
        write_param_line(&mut out, "ldc2_4", &self.ldc2_4);
        write_param_line(&mut out, "velocity_scale", &self.velocity_scale);
        write_param_line(&mut out, "beam_factor1", &self.beam_factor1);
        write_param_line(&mut out, "beam_factor2", &self.beam_factor2);
        write_param_line(&mut out, "t0", &self.t0);
        write_param_line(&mut out, "period", &self.period);
        write_param_line(&mut out, "pdot", &self.pdot);
        write_param_line(&mut out, "deltat", &self.deltat);
        write_param_line(&mut out, "gravity_dark1", &self.gravity_dark1);
        write_param_line(&mut out, "gravity_dark2", &self.gravity_dark2);
        write_param_line(&mut out, "absorb", &self.absorb);
        write_param_line(&mut out, "slope", &self.slope);
        write_param_line(&mut out, "quad", &self.quad);
        write_param_line(&mut out, "cube", &self.cube);
        write_param_line(&mut out, "third", &self.third);
        write_param_line(&mut out, "rdisc1", &self.rdisc1);
        write_param_line(&mut out, "rdisc2", &self.rdisc2);
        write_param_line(&mut out, "height_disc", &self.height_disc);
        write_param_line(&mut out, "beta_disc", &self.beta_disc);
        write_param_line(&mut out, "temp_disc", &self.temp_disc);
        write_param_line(&mut out, "texp_disc", &self.texp_disc);
        write_param_line(&mut out, "lin_limb_disc", &self.lin_limb_disc);
        write_param_line(&mut out, "quad_limb_disc", &self.quad_limb_disc);
        write_param_line(&mut out, "temp_edge", &self.temp_edge);
        write_param_line(&mut out, "absorb_edge", &self.absorb_edge);
        write_param_line(&mut out, "radius_spot", &self.radius_spot);
        write_param_line(&mut out, "length_spot", &self.length_spot);
        write_param_line(&mut out, "height_spot", &self.height_spot);
        write_param_line(&mut out, "expon_spot", &self.expon_spot);
        write_param_line(&mut out, "epow_spot", &self.epow_spot);
        write_param_line(&mut out, "angle_spot", &self.angle_spot);
        write_param_line(&mut out, "yaw_spot", &self.yaw_spot);
        write_param_line(&mut out, "temp_spot", &self.temp_spot);
        write_param_line(&mut out, "tilt_spot", &self.tilt_spot);
        write_param_line(&mut out, "cfrac_spot", &self.cfrac_spot);
        write_param_line(&mut out, "stsp11_long", &self.stsp11_long);
        write_param_line(&mut out, "stsp11_lat", &self.stsp11_lat);
        write_param_line(&mut out, "stsp11_fwhm", &self.stsp11_fwhm);
        write_param_line(&mut out, "stsp11_tcen", &self.stsp11_tcen);
        write_param_line(&mut out, "stsp12_long", &self.stsp12_long);
        write_param_line(&mut out, "stsp12_lat", &self.stsp12_lat);
        write_param_line(&mut out, "stsp12_fwhm", &self.stsp12_fwhm);
        write_param_line(&mut out, "stsp12_tcen", &self.stsp12_tcen);
        write_param_line(&mut out, "stsp13_long", &self.stsp13_long);
        write_param_line(&mut out, "stsp13_lat", &self.stsp13_lat);
        write_param_line(&mut out, "stsp13_fwhm", &self.stsp13_fwhm);
        write_param_line(&mut out, "stsp13_tcen", &self.stsp13_tcen);
        write_param_line(&mut out, "stsp21_long", &self.stsp21_long);
        write_param_line(&mut out, "stsp21_lat", &self.stsp21_lat);
        write_param_line(&mut out, "stsp21_fwhm", &self.stsp21_fwhm);
        write_param_line(&mut out, "stsp21_tcen", &self.stsp21_tcen);
        write_param_line(&mut out, "stsp22_long", &self.stsp22_long);
        write_param_line(&mut out, "stsp22_lat", &self.stsp22_lat);
        write_param_line(&mut out, "stsp22_fwhm", &self.stsp22_fwhm);
        write_param_line(&mut out, "stsp22_tcen", &self.stsp22_tcen);
        write_param_line(&mut out, "uesp_long1", &self.uesp_long1);
        write_param_line(&mut out, "uesp_long2", &self.uesp_long2);
        write_param_line(&mut out, "uesp_lathw", &self.uesp_lathw);
        write_param_line(&mut out, "uesp_taper", &self.uesp_taper);
        write_param_line(&mut out, "uesp_temp", &self.uesp_temp);
        write_f64_line(&mut out, "delta_phase", self.delta_phase);
        write_u32_line(&mut out, "nlat1f", self.nlat1f);
        write_u32_line(&mut out, "nlat2f", self.nlat2f);
        write_u32_line(&mut out, "nlat1c", self.nlat1c);
        write_u32_line(&mut out, "nlat2c", self.nlat2c);
        write_bool_line(&mut out, "npole", self.npole);
        write_u32_line(&mut out, "nlatfill", self.nlatfill);
        write_u32_line(&mut out, "nlngfill", self.nlngfill);
        write_f64_line(&mut out, "lfudge", self.lfudge);
        write_f64_line(&mut out, "llo", self.llo);
        write_f64_line(&mut out, "lhi", self.lhi);
        write_f64_line(&mut out, "phase1", self.phase1);
        write_f64_line(&mut out, "phase2", self.phase2);
        write_f64_line(&mut out, "wavelength", self.wavelength);
        write_bool_line(&mut out, "roche1", self.roche1);
        write_bool_line(&mut out, "roche2", self.roche2);
        write_bool_line(&mut out, "eclipse1", self.eclipse1);
        write_bool_line(&mut out, "eclipse2", self.eclipse2);
        write_bool_line(&mut out, "glens1", self.glens1);
        write_bool_line(&mut out, "use_radii", self.use_radii);
        write_f64_line(&mut out, "tperiod", self.tperiod);
        write_bool_line(&mut out, "gdark_bolom1", self.gdark_bolom1);
        write_bool_line(&mut out, "gdark_bolom2", self.gdark_bolom2);
        write_f64_line(&mut out, "mucrit1", self.mucrit1);
        write_f64_line(&mut out, "mucrit2", self.mucrit2);
        write_ldc_line(&mut out, "limb1", self.limb1);
        write_ldc_line(&mut out, "limb2", self.limb2);
        write_bool_line(&mut out, "mirror", self.mirror);
        write_bool_line(&mut out, "add_disc", self.add_disc);
        write_u32_line(&mut out, "nrad", self.nrad);
        write_bool_line(&mut out, "opaque", self.opaque);
        write_bool_line(&mut out, "add_spot", self.add_spot);
        write_u32_line(&mut out, "nspot", self.nspot);
        write_bool_line(&mut out, "iscale", self.iscale);
        write(path, out).map_err(|e| e.to_string())
    }

    pub fn basic_defined(&self) -> bool {
        self.q.defined
        && self.iangle.defined
        && ((self.r1.defined && self.r2.defined && self.use_radii)
            || (self.cphi3.defined && self.cphi4.defined && !self.use_radii))
        && self.t1.defined
        && self.t2.defined
        && self.ldc1_1.defined
        && self.ldc2_1.defined
        && self.t0.defined
        && self.period.defined
        && self.gravity_dark1.defined
        && self.gravity_dark2.defined
        && self.absorb.defined
    }

    pub fn disc_defined(&self) -> bool {
        self.rdisc1.defined
        && self.rdisc2.defined
        && self.height_disc.defined
        && self.beta_disc.defined
        && self.temp_disc.defined
        && self.texp_disc.defined
        && self.lin_limb_disc.defined
    }

    pub fn bright_spot_defined(&self) -> bool {
        self.radius_spot.defined
        && self.length_spot.defined
        && self.height_spot.defined
        && self.expon_spot.defined
        && self.epow_spot.defined
        && self.angle_spot.defined
        && self.yaw_spot.defined
        && self.temp_spot.defined
        && self.tilt_spot.defined
        && self.cfrac_spot.defined
    }

    pub fn validate(&self) -> Result<(), RocheError> {
        const MAX_TEMP: f64 = 1.0e10;

        if !self.basic_defined() {
            return Err(RocheError::ParameterError("Not all necessary parameters are defined.".to_string()));
        }
        check_parameter_f64_or_error(self.iangle.value, 0.0, 90.0, "iangle must be between 0.0 and 90.0.")?;
        check_parameter_f64_or_error(self.q.value, f64::MIN_POSITIVE, f64::MAX, "q must be positive and non-zero.")?;
        let rl1: f64 = x_l1_1(self.q.value, self.spin1.value).unwrap();
        let rl2: f64 = 1.0 - x_l1_2(self.q.value, self.spin2.value).unwrap();

        if self.use_radii {
            check_parameter_f64_or_error(self.r1.value, f64::MIN_POSITIVE, rl1, "r1 must be between 0.0 and 1.0 and not exceed its Roche lobe.")?;
            check_parameter_f64_or_error(self.r2.value, -1.0, rl2, "r2 must be between -1.0 and 1.0 and not exceed its Roche lobe.")?;
        } else {
            let (r1, r2) = self.get_r1r2();
            check_parameter_f64_or_error(r1, 0.0, rl1, "cphi3 and cphi4 must correspond to a primary radius between 0.0 and 1.0 that does not exceed its Roche lobe.")?;
            check_parameter_f64_or_error(r2, 0.0, rl2, "cphi3 and cphi4 must correspond to a secondary radius between 0.0 and 1.0 that does not exceed its Roche lobe.")?;
        }
        check_parameter_f64_or_error(self.t1.value, 0.0, MAX_TEMP, "t1 must be positive.")?;
        check_parameter_f64_or_error(self.t2.value, 0.0, MAX_TEMP, "t2 must be positive.")?;
        check_parameter_f64_or_error(self.velocity_scale.value, f64::MIN_POSITIVE, C / 1000.0, "velocity_scale must be positive and not exceed the speed of light.")?;
        check_parameter_f64_or_error(self.period.value, f64::MIN_POSITIVE, f64::INFINITY, "period must be positive and non-zero.")?;
        check_parameter_f64_or_error(self.absorb.value, 0.0, f64::MAX, "absorb must be positive")?;
        check_parameter_f64_or_error(self.third.value, 0.0, f64::MAX, "third must be positive")?;
        check_parameter_f64_or_error(self.wavelength, f64::MIN_POSITIVE, f64::MAX, "wavelength must be positive and non-zero.")?; 
        check_parameter_f64_or_error(self.tperiod, f64::MIN_POSITIVE, f64::MAX, "tperiod must be positive and non-zero.")?;

        if self.add_disc {
            if !self.disc_defined(){
               return Err(RocheError::ParameterError("Necessary disc parameters not defined.".to_string())); 
            }
            check_parameter_f64_or_error(self.rdisc1.value, -1.0, rl1, "rdisc1 must be between -1.0 and 1.0 and not exceed the Roche lobe of the primary star.")?;
            check_parameter_f64_or_error(self.rdisc2.value, -1.0, rl1, "rdisc2 must be between -1.0 and 1.0 and not exceed the Roche lobe of the primary star.")?;
            check_parameter_f64_or_error(self.temp_disc.value, 0.0, MAX_TEMP, "temp_disc must be positive.")?;
            check_parameter_f64_or_error(self.temp_edge.value, 0.0, MAX_TEMP, "temp_edge must be positive.")?;
            check_parameter_f64_or_error(self.absorb_edge.value, 0.0, f64::MAX, "absorb_edge must be positive.")?;
        }

        if self.add_spot {
            if !self.bright_spot_defined(){
                return Err(RocheError::ParameterError("Not all necessary bright spot parameters are defined.".to_string()));
            }
            check_parameter_f64_or_error(self.radius_spot.value, f64::MIN_POSITIVE, 1.0, "radius_spot must be between 0.0 and 1.0 and not exceed the Roche lobe of the primary star.")?;
            check_parameter_f64_or_error(self.temp_spot.value, 0.0, MAX_TEMP, "temp_spot must be positive.")?;
        }
        
        Ok(())

    }

    pub fn get_r1r2(&self) -> (f64, f64) {
        if self.use_radii {
            (self.r1.value, self.r2.value)
        } else {
            let sini = self.iangle.value.to_radians().sin();
            let r2pr1 = (1. - (sini * (2. * PI * self.cphi4.value).cos()).powi(2)).sqrt();
            let r2mr1 = (1. - (sini * (2. * PI * self.cphi3.value).cos()).powi(2)).sqrt();
            let rr1 = (r2pr1 - r2mr1) / 2.;
            let rr2 = (r2pr1 + r2mr1) / 2.;
            (rr1, rr2)
        }
    }

    pub fn get_ldc1(&self) -> LDC {
        LDC::with_params(
            self.ldc1_1.value,
            self.ldc1_2.value,
            self.ldc1_3.value,
            self.ldc1_4.value,
            self.mucrit1,
            self.limb1,
        )
    }

    pub fn get_ldc2(&self) -> LDC {
        LDC::with_params(
            self.ldc2_1.value,
            self.ldc2_2.value,
            self.ldc2_3.value,
            self.ldc2_4.value,
            self.mucrit2,
            self.limb2,
        )
    }

    pub fn apply_update(&mut self, updated_model: ModelUpdate) -> Result<(), RocheError> {
        updated_model.check_bounds()?;
        apply_update!(self, updated_model, {
            q: pparam,
            iangle: pparam,
            r1: pparam,
            r2: pparam,
            cphi3: pparam,
            cphi4: pparam,
            spin1: pparam,
            spin2: pparam,
            t1: pparam,
            t2: pparam,
            ldc1_1: pparam,
            ldc1_2: pparam,
            ldc1_3: pparam,
            ldc1_4: pparam,
            ldc2_1: pparam,
            ldc2_2: pparam,
            ldc2_3: pparam,
            ldc2_4: pparam,
            velocity_scale: pparam,
            beam_factor1: pparam,
            beam_factor2: pparam,
            t0: pparam,
            period: pparam,
            pdot: pparam,
            deltat: pparam,
            gravity_dark1: pparam,
            gravity_dark2: pparam,
            absorb: pparam,
            slope: pparam,
            quad: pparam,
            cube: pparam,
            third: pparam,
            rdisc1: pparam,
            rdisc2: pparam,
            height_disc: pparam,
            beta_disc: pparam,
            temp_disc: pparam,
            texp_disc: pparam,
            lin_limb_disc: pparam,
            quad_limb_disc: pparam,
            temp_edge: pparam,
            absorb_edge: pparam,
            radius_spot: pparam,
            length_spot: pparam,
            height_spot: pparam,
            expon_spot: pparam,
            epow_spot: pparam,
            angle_spot: pparam,
            yaw_spot: pparam,
            temp_spot: pparam,
            tilt_spot: pparam,
            cfrac_spot: pparam,
            stsp11_long: pparam,
            stsp11_lat: pparam,
            stsp11_fwhm: pparam,
            stsp11_tcen: pparam,
            stsp12_long: pparam,
            stsp12_lat: pparam,
            stsp12_fwhm: pparam,
            stsp12_tcen: pparam,
            stsp13_long: pparam,
            stsp13_lat: pparam,
            stsp13_fwhm: pparam,
            stsp13_tcen: pparam,
            stsp21_long: pparam,
            stsp21_lat: pparam,
            stsp21_fwhm: pparam,
            stsp21_tcen: pparam,
            stsp22_long: pparam,
            stsp22_lat: pparam,
            stsp22_fwhm: pparam,
            stsp22_tcen: pparam,
            uesp_long1: pparam,
            uesp_long2: pparam,
            uesp_lathw: pparam,
            uesp_taper: pparam,
            uesp_temp: pparam,
            delta_phase: plain,
            nlat1f: plain,
            nlat2f: plain,
            nlat1c: plain,
            nlat2c: plain,
            npole: plain,
            nlatfill: plain,
            nlngfill: plain,
            lfudge: plain,
            llo: plain,
            lhi: plain,
            phase1: plain,
            phase2: plain,
            wavelength: plain,
            roche1: plain,
            roche2: plain,
            eclipse1: plain,
            eclipse2: plain,
            glens1: plain,
            use_radii: plain,
            tperiod: plain,
            gdark_bolom1: plain,
            gdark_bolom2: plain,
            mucrit1: plain,
            mucrit2: plain,
            limb1: enum,
            limb2: enum,
            mirror: plain,
            add_disc: plain,
            nrad: plain,
            opaque: plain,
            add_spot: plain,
            nspot: plain,
            iscale: plain
        });
    Ok(())
    }
}

#[pymethods]
impl Model {

    #[new]
    fn new() -> Self {
        Model::default()
    }

    #[staticmethod]
    #[pyo3(name="from_file")]
    fn from_file_py(path: &str) -> PyResult<Self> {
        Model::from_file(path).map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[pyo3(name="write")]
    fn write_py(&self, path: &str) -> PyResult<()> {
        self.write(path)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    fn update(&mut self, _py: Python, dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let upd: ModelUpdate = from_pyobject(dict.clone())?;
        self.apply_update(upd)?;
        Ok(())
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(serde_pyobject::to_pyobject(py, self)?)
    }

    fn __repr__(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __getstate__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        serde_pyobject::to_pyobject(py, self)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            })
    }

    fn __setstate__(
        &mut self,
        state: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let restored: Model =
            serde_pyobject::from_pyobject(state.clone())
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(e.to_string())
                })?;

        *self = restored;
        Ok(())
    }
}

fn default_pparam() -> Pparam {
    Pparam::default()
}

fn default_delta_phase() -> f64 {
    1.0e-7_f64
}

fn default_zero() -> u32 {
    0_u32
}

fn default_zero_f64() -> f64 {
    0.0_f64
}

fn default_ten() -> u32 {
    10_u32
}

fn default_true() -> bool {
    true
}

fn default_poly() -> LDCType {
    LDCType::Poly
}

fn default_false() -> bool {
    false
}

fn parse_entry(line: &str) -> Option<(String, Entry)> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    let mut it = line.split_whitespace();

    let name = it.next()?.to_string();
    if it.next()? != "=" {
        return None;
    }

    let rest: Vec<_> = it.collect();

    match rest.len() {
        1 => Some((name, Entry::Scalar(rest[0].to_string()))),
        5 => {
            let joined = format!("{} = {}", name, rest.join(" "));
            let p: Pparam = joined.parse().ok()?;
            Some((name, Entry::Param(p)))
        }
        _ => None,
    }
}

fn read_lines<P: AsRef<Path>>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>> {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn load_entries(path: &str) -> Result<HashMap<String, Entry>, String> {
    let mut map = HashMap::new();

    for line in read_lines(path)
        .map_err(|e| e.to_string())?
        .map_while(Result::ok)
    {
        if let Some((name, entry)) = parse_entry(&line) {
            map.insert(name, entry);
        }
    }

    Ok(map)
}

fn get_p(map: &HashMap<String, Entry>, k: &str) -> Result<Pparam, String> {
    match map.get(k) {
        Some(Entry::Param(p)) => Ok(*p),
        _ => Err(format!("missing Pparam: {}", k)),
    }
}

fn get_f64(map: &HashMap<String, Entry>, k: &str) -> Result<f64, String> {
    match map.get(k) {
        Some(Entry::Scalar(v)) => v.parse().map_err(|_| format!("bad f64: {}", k)),
        _ => Err(format!("missing f64: {}", k)),
    }
}

fn get_u32(map: &HashMap<String, Entry>, k: &str) -> Result<u32, String> {
    match map.get(k) {
        Some(Entry::Scalar(v)) => v.parse().map_err(|_| format!("bad u32: {}", k)),
        _ => Err(format!("missing u32: {}", k)),
    }
}

fn get_bool(map: &HashMap<String, Entry>, k: &str) -> Result<bool, String> {
    match map.get(k) {
        Some(Entry::Scalar(v)) => match v.as_str() {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(format!("bad bool: {}", k)),
        },
        _ => Err(format!("missing bool: {}", k)),
    }
}

fn get_ldc(map: &HashMap<String, Entry>, k: &str) -> Result<LDCType, String> {
    match map.get(k) {
        Some(Entry::Scalar(v)) => match v.as_str() {
            "Claret" => Ok(LDCType::Claret),
            "Poly" => Ok(LDCType::Poly),
            _ => Err(format!("bad LDCType: {}", k)),
        },
        _ => Err(format!("missing LDCType: {}", k)),
    }
}

fn pparam_update_to_value(pparam_update: PparamUpdate) -> Option<f64> {
    match pparam_update {
        PparamUpdate::Full(pparam) => Some(pparam.value),
        PparamUpdate::Partial(partial) => partial.value,
        PparamUpdate::Value(value) => Some(value),
    }
}

fn check_pparam_update(pparam_update: PparamUpdate, lower_limit: f64, upper_limit: f64) -> bool {
    let Some(value) = pparam_update_to_value(pparam_update) else {
        return true
    };
    check_parameter_f64(value, lower_limit, upper_limit)
}

fn check_pparam_update_or_error(pparam_update: PparamUpdate, lower_limit: f64, upper_limit: f64, error_msg: &str) -> Result<(), RocheError> {
    if !check_pparam_update(pparam_update, lower_limit, upper_limit) {
        return Err(RocheError::ParameterError(error_msg.to_string()));
    }
    Ok(())
}

fn check_parameter_f64(value: f64, lower_limit: f64, upper_limit: f64) -> bool {
    value >= lower_limit && value <= upper_limit
}

fn check_parameter_f64_or_error(value: f64, lower_limit: f64, upper_limit: f64, error_msg: &str) -> Result<(), RocheError> {
    if !check_parameter_f64(value, lower_limit, upper_limit) {
        return Err(RocheError::ParameterError(error_msg.to_string()));
    }
    Ok(())
}

fn write_param_line(out: &mut String, name: &str, p: &Pparam) {
    out.push_str(&format!("{:<15} = {}\n", name, p));
}

fn write_f64_line(out: &mut String, name: &str, v: f64) {
    out.push_str(&format!("{:<15} = {}\n", name, v));
}

fn write_u32_line(out: &mut String, name: &str, v: u32) {
    out.push_str(&format!("{:<15} = {}\n", name, v));
}

fn write_bool_line(out: &mut String, name: &str, v: bool) {
    let val = if v { 1 } else { 0 };
    out.push_str(&format!("{:<15} = {}\n", name, val));
}

fn write_ldc_line(out: &mut String, name: &str, v: LDCType) {
    let s = match v {
        LDCType::Claret => "Claret",
        LDCType::Poly => "Poly",
    };
    out.push_str(&format!("{:<15} = {}\n", name, s));
}
