|PyPI version shields.io|

.. |PyPI version shields.io| image:: https://img.shields.io/pypi/v/lcurve.svg
    :target: https://pypi.python.org/pypi/lcurve/


Python extension replicating Tom Marsh's LCURVE (cpp-lcurve) translated into Rust
===================================


.. code-block:: python

    import lcurve

    binary_model = lcurve.BinaryModel.from_file("WDdM.mod")

load in your data in python and then convert each column to a contiguous array.

.. code-block:: python

    time, t_exp, flux, flux_err, weight, n_div = np.loadtxt("example_data.dat", unpack=True)
    time = np.ascontiguousarray(time)
    t_exp = np.ascontiguousarray(t_exp)
    flux = np.ascontiguousarray(flux)
    flux_err = np.ascontiguousarray(flux_err)
    weight = np.ascontiguousarray(weight)
    n_div = np.ascontiguousarray(n_div)


Then calculate the model. Note that flux, flux_err, weight, and scale_factor are optional arguments. If flux and flux_err are supplied and scale_factor is not then the returned light curve model will be scaled to minimise the chi-squared. If weights are not given they will be assumed to be 1.
In all cases the scale_factor that is used can be accessed as an attribute.

.. code-block:: python

    lc = binary_model.compute_light_curve(time, t_exp, n_div, flux, flux_err, weight, scale_factor=100000.0)

    print(lc.scale_factor)

    fig, ax = plt.subplots()
    ax.plot(time, lc.star1)
    ax.plot(time, lc.star2)
    ax.plot(time, lc.disc)
    ax.plot(time, lc.disc_edge)
    ax.plot(time, lc.bright_spot)
    ax.plot(time, lc.total)
    plt.show()

Volume-averaged radii and flux-weighted log(g) calculated from the model can be accessed as attributes (Note: The log(g) values will be `None` if the velocity_scale is not 'defined').

.. code-block:: python

    print(lc.rva1)
    print(lc.rva2)
    print(lc.logg1)
    print(lc.logg2)

And if arrays of flux and flux_err were given then the chi-squared and log(probability) can also be accessed

.. code-block:: python

    print(lc.chi2)
    print(lc.log_prob)

If flux and flux_err are not given then these will both be `None`.

Once initialised, the model can be updated from a python dictionary and also accessed as a python dictionary.

.. code-block:: python

    model.update({"iangle": 85.5, "t1": 20000.0})
    current_parameters = binary_model.model.to_dict()

If you are updating a parameter that was not previously 'defined', then 'defined' will automatically  be set to 'true' when updated.
If you want to set it back to 'false' then the whole parameter details must be given e.g.

.. code-block:: python

    binary_model.update({"radius_spot": {"value": 0.326, "range": 0.0, "dstep": 0.0, "vary": False, "defined": False}})

Note that `model.update` checks if the grid needs to be rebuilt depending on what parameters are given. This is useful for fitting multiband data where the grid will remain the same for all bands but the fluxes and wavelengths will change, saving time by only updating the fluxes of the gridpoints.

Checks are now performed to make sure parameters are within allowed ranges when updating from a python dictionary using `BinaryModel.update` or `model.update`. Validation checks are also carried out before the grid is built or updated but it's possible that some edge-cases may exist that don't get flagged so remain cautious about the parameters you supply.

The LCURVE fitting routines, `levmarq`, and `simplex` have not been implemented in lcurve, however the relatively simple python interface means that fitting algorithms from scipy can be used easily.

.. code-block:: python

    from scipy.minimize import curve_fit, minimize


    def fit_function_minimize(params):
        model.update({"t0": params[0]})
        return model.compute_light_curve(time, exp, n_div, flux, flux_err, weight).chi2


    res = minimize(fit_function_minimize, [58674.006198], method='Nelder-Mead')
    print(f"t0 = {res.x[0]}")


    def fit_function_curve_fit(time, t0):
        model.update({"t0": t0})
        return model.compute_light_curve(time, exp, n_div, flux, flux_err, weight).total

    popt, pcov = curve_fit(fit_function_curve_fit, xdata=time, ydata=flux, p0=[58674.006198], sigma=flux_err, method='lm', xtol=1e-12)
    print(f"t0 = {popt[0]} +- {np.sqrt(np.diag(pcov))[0]}")
    
It is also possible to supply fluxes directly to the surface grids using a numpy array so that light curve models can be calculated for a given surface flux map. For this, the grids must first be initialised as normal before updating the flux of the grid points.

.. code-block:: python

    binary_model = lcurve.BinaryModel.from_file("WDdM.mod")
    star1f = binary_model.star1_fine_grid
    binary_model.set_grid_fluxes("star1_fine", 1.0e-10*np.ones_like(star1f))

To be able to match up a given surface flux map with the grid, you will need to access the position of each point on the grid (it's up to you to work out how best to interpolate a flux map onto the grid positions!).
e.g.:

.. code-block:: python

    positions = np.array([point.position for point in binary_model.star1_fine_grid])

Note that lcurve defines the origin, Vec3(0, 0, 0), as the centre-of-mass of star1 and the centre-of-mass of star2 at the position, Vec3(1, 0, 0).
When defining grid-fluxes like this, the coarse and fine grids will be scaled to give the same total flux at the phases of the switching points as is the case with normally defined grids. However it's probably best to set the fine and coarse grids to the same number of points to prevent any possible issues here.


Output convergence with C++
===========================
To run the tests that check convergence with _legacy_ C++ code, run

.. code-block:: bash

   pytest

from the root of the repository.
