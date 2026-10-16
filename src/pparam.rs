use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::fmt;
use pyo3::prelude::*;

#[pyclass(from_py_object, module = "lcurve")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pparam {
    #[pyo3(get, set)]
    pub value: f64,
    #[pyo3(get, set)]
    pub range: f64,
    #[pyo3(get, set)]
    pub dstep: f64,
    #[pyo3(get, set)]
    pub vary: bool,
    #[pyo3(get, set)]
    pub defined: bool,
}

#[pymethods]
impl Pparam {

    #[new]
    pub fn new(value: f64, range: f64, dstep: f64, vary: bool, defined: bool) -> Self {
        Self { value, range, dstep, vary, defined }
    }

    fn __repr__(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __getnewargs__(&self) -> (f64, f64, f64, bool, bool) {
        (
            self.value,
            self.range,
            self.dstep,
            self.vary,
            self.defined,
        )
    }

}

impl Default for Pparam {
    
    fn default() -> Self {
        Self {
            value: 0.0,
            range: 0.0,
            dstep: 0.0,
            vary: false,
            defined: false,
        }
    }
}

impl FromStr for Pparam {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut fields = s.split_whitespace();
        let _name: &str = fields.next().ok_or("missing value")?;
        let _equals: &str = fields.next().ok_or("missing value")?;
        let value = fields
            .next()
            .ok_or("missing value")?
            .parse()
            .map_err(|_| "bad value")?;
        let range = fields
            .next()
            .ok_or("missing range")?
            .parse()
            .map_err(|_| "bad range")?;
        let dstep = fields
            .next()
            .ok_or("missing dstep")?
            .parse()
            .map_err(|_| "bad dstep")?;
        let vary = fields
            .next()
            .ok_or("missing vary")?
            .parse::<i32>()
            .map_err(|_| "bad vary")?
            != 0;
        let defined = fields
            .next()
            .ok_or("missing vary")?
            .parse::<i32>()
            .map_err(|_| "bad vary")?
            != 0;

        Ok(Pparam {
            value,
            range,
            dstep,
            vary,
            defined,
        })
    }
}

impl fmt::Display for Pparam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vary = if self.vary { 1 } else { 0 };
        let defined = if self.defined { 1 } else { 0 };
        write!(
            f,
            "{} {} {} {} {}",
            self.value,
            self.range,
            self.dstep,
            vary,
            defined
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PparamPartial {
    pub value: Option<f64>,
    pub range: Option<f64>,
    pub dstep: Option<f64>,
    pub vary: Option<bool>,
    pub defined: Option<bool>,
}
