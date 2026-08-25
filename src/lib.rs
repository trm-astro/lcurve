use pyo3::prelude::*;

pub mod binary_model;
pub mod comp_gravity;
pub mod comp_light;
pub mod comp_radius;
pub mod ginterp;
pub mod ldc;
pub mod model;
pub mod numface;
pub mod pparam;
pub mod set_bright_spot_grid;
pub mod set_disc_continuum;
pub mod set_disc_grid;
pub mod set_star_continuum;
pub mod set_star_grid;

#[pymodule]
mod lcurve {

    #[pymodule_export]
    use crate::binary_model::BinaryModel;

    #[pymodule_export]
    use crate::model::Model;

    #[pymodule_export]
    use crate::pparam::Pparam;

    #[pymodule_export]
    use crate::ldc::LDCType;
    
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");
}
