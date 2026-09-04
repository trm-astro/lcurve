use roche::{self, Point, Vec3};
use std::f64::consts::TAU;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// 
/// Grid is a struct to hold the vector of `Vec<Points>` defining a
/// component grid along with methods to act on this.
/// 
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Grid {
    #[pyo3(get)]
    pub points: Vec<Point>,
}

impl Grid {

    pub fn new(points: Vec<Point>) -> Self {
        Self {
            points,
        }
    }

    pub fn position(&self, iangle: f64, phase: Option<f64>) -> Vec<Vec3> {
        
        let position: Vec<Vec3> = match phase {
            Some(phase) => {
                let mut position: Vec<Vec3> = vec![];
                let earth = roche::set_earth_iangle(iangle, phase);
                for point in &self.points {
                    if earth.dot(&point.direction) > 0.0 && point.is_visible(phase) {
                        position.push(point.position);
                    }
                }
                position
            },
            None => {
                let mut position: Vec<Vec3> = vec![];
                for point in &self.points {
                    position.push(point.position);
                }
                position
            }
        };

        position
    }

    pub fn direction(&self, iangle: f64, phase: Option<f64>) -> Vec<Vec3> {
        
        let direction: Vec<Vec3> = match phase {
            Some(phase) => {
                let mut direction: Vec<Vec3> = vec![];
                let earth = roche::set_earth_iangle(iangle, phase);
                for point in &self.points {
                    if earth.dot(&point.direction) > 0.0 && point.is_visible(phase) {
                        direction.push(point.direction);
                    }
                }
                direction
            },
            None => {
                let mut direction: Vec<Vec3> = vec![];
                for point in &self.points {
                    direction.push(point.direction);
                }
                direction
            }
        };

        direction
    }

    pub fn gravity(&self, iangle: f64, phase: Option<f64>) -> Vec<f32> {
        
        let gravity: Vec<f32> = match phase {
            Some(phase) => {
                let mut gravity: Vec<f32> = vec![];
                let earth = roche::set_earth_iangle(iangle, phase);
                for point in &self.points {
                    if earth.dot(&point.direction) > 0.0 && point.is_visible(phase) {
                        gravity.push(point.gravity);
                    }
                }
                gravity
            },
            None => {
                let mut gravity: Vec<f32> = vec![];
                for point in &self.points {
                    gravity.push(point.gravity);
                }
                gravity
            }
        };

        gravity
    }

    pub fn area(&self, iangle: f64, phase: Option<f64>) -> Vec<f32> {
        
        let area: Vec<f32> = match phase {
            Some(phase) => {
                let mut area: Vec<f32> = vec![];
                let earth = roche::set_earth_iangle(iangle, phase);
                for point in &self.points {
                    if earth.dot(&point.direction) > 0.0 && point.is_visible(phase) {
                        area.push(point.area);
                    }
                }
                area
            },
            None => {
                let mut area: Vec<f32> = vec![];
                for point in &self.points {
                    area.push(point.area);
                }
                area
            }
        };

        area
    }

    pub fn flux(&self, iangle: f64, phase: Option<f64>) -> Vec<f32> {
        
        let flux: Vec<f32> = match phase {
            Some(phase) => {
                let mut flux: Vec<f32> = vec![];
                let earth = roche::set_earth_iangle(iangle, phase);
                for point in &self.points {
                    if earth.dot(&point.direction) > 0.0 && point.is_visible(phase) {
                        flux.push(point.flux);
                    }
                }
                flux
            },
            None => {
                let mut flux: Vec<f32> = vec![];
                for point in &self.points {
                    flux.push(point.flux);
                }
                flux
            }
        };

        flux
    }

}

#[pymethods]
impl Grid {

    #[pyo3(name="area", signature = (iangle, phase=None))]
    pub fn python_area(&self, py: Python, iangle: f64, phase: Option<f64>) -> Py<PyArray1<f32>> {
        
        let area: Vec<f32> = self.area(iangle, phase);
        area.into_pyarray(py).unbind()
    }

    #[pyo3(name="flux", signature = (iangle, phase=None))]
    pub fn python_flux(&self, py: Python, iangle: f64, phase: Option<f64>) -> Py<PyArray1<f32>> {
        
        let flux: Vec<f32> = self.flux(iangle, phase);
        flux.into_pyarray(py).unbind()
    }

    ///
    /// Projects the grid onto a 2D plane as seen at the model inclination
    /// at the supplied phase with the binary centre of mass as the origin.
    /// 
    /// Arguments
    /// * `q` - Binary mass ratio M2/M1.
    /// * `iangle` - Orbital inclination at which to project the grid.
    /// * `phase` - Orbital phase at which to project the grid
    /// 
    /// Returns
    /// (x, y) - Arrays of projected grid point positions
    /// 
    #[pyo3(signature = (q, iangle, phase))]
    pub fn project_2d(&self, q: f64, iangle: f64, phase: f64) -> (Vec<f64>, Vec<f64>) {

        let mut x_arr: Vec<f64> = vec![];
        let mut y_arr: Vec<f64> = vec![];
        let cofm = Vec3::new(q/(1.0+q), 0.0, 0.0);
        let (sinp, cosp) = (TAU*phase).sin_cos();
        let earth = roche::set_earth_iangle(iangle, phase);
        let xsky = Vec3::new(sinp, cosp, 0.0);
        let ysky = earth.cross(&xsky);
        
        for point in &self.points {
            if earth.dot(&point.direction) > 0.0 && point.is_visible(phase) {
                let r = point.position - cofm;
                x_arr.push(r.dot(&xsky));
                y_arr.push(r.dot(&ysky));
            }
        }
        (x_arr, y_arr)
    }
    
}
