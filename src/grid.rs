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
        self.visible_values(iangle, phase, |point| point.position)
    }

    pub fn direction(&self, iangle: f64, phase: Option<f64>) -> Vec<Vec3> {
        self.visible_values(iangle, phase, |point| point.direction)
    }

    pub fn area(&self, iangle: f64, phase: Option<f64>) -> Vec<f32> {
        self.visible_values(iangle, phase, |point| point.area)
    }
    
    pub fn gravity(&self, iangle: f64, phase: Option<f64>) -> Vec<f32> {
        self.visible_values(iangle, phase, |point| point.gravity)
    }

    pub fn eclipse(&self, iangle: f64, phase: Option<f64>) -> Vec<Vec<(f64, f64)>> {
        self.visible_values(iangle, phase, |point| point.eclipse.clone())
    }

    pub fn flux(&self, iangle: f64, phase: Option<f64>) -> Vec<f32> {
        self.visible_values(iangle, phase, |point| point.flux)
    }

    /// 
    /// Given an inclination, a function, and an optional phase, visible_values
    /// filters the points based on the visibility and returns the filtered
    /// output of the supplied function.
    /// 
    fn visible_values<T, F>(&self, iangle: f64, phase: Option<f64>, f: F) -> Vec<T>
    where F: Fn(&Point) -> T {
        match phase {
            Some(phase) => {
                let earth = roche::set_earth_iangle(iangle, phase);

                self.points
                    .iter()
                    .filter(|point| {
                        earth.dot(&point.direction) > 0.0
                            && point.is_visible(phase)
                    })
                    .map(f)
                    .collect()
            }
            None => self.points
                        .iter()
                        .map(f)
                        .collect()
        }
    }

}

#[pymethods]
impl Grid {

    #[pyo3(name="area", signature = (iangle, phase=None))]
    pub fn python_area(&self, py: Python, iangle: f64, phase: Option<f64>) -> Py<PyArray1<f32>> {
        
        let area: Vec<f32> = self.area(iangle, phase);
        area.into_pyarray(py).unbind()
    }

    #[pyo3(name="gravity", signature = (iangle, phase=None))]
    pub fn python_gravity(&self, py: Python, iangle: f64, phase: Option<f64>) -> Py<PyArray1<f32>> {
        
        let gravity: Vec<f32> = self.gravity(iangle, phase);
        gravity.into_pyarray(py).unbind()
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
