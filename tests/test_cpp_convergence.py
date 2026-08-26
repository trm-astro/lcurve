from __future__ import annotations

import os
import pytest
import numpy as np

import lcurve


# Setup function. If everything goes well, this should not be modified.
def fetch_object_files() -> list[tuple[str, str]]:
    # Make a list of all directories that are in a given directory, omitting __pycache__
    list_directories_full_path = lambda path: [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d != "__pycache__"
    ]

    # Do the above for the `./tests/` directory
    files = list_directories_full_path("./tests/")

    # Make tuples of modfile cpp_solution file combinations for each directory
    return [(dir + "/modfile.mod", dir + "/cpp_solution.dat") for dir in files]


@pytest.mark.parametrize("modfile, cpp_solution_file", fetch_object_files())
def test_compare_cpp_rust_solution(modfile: str, cpp_solution_file: str) -> None:

    # Load the mod file and the solution
    binary_model = lcurve.BinaryModel.from_file(modfile)
    cpp_solution = np.loadtxt(cpp_solution_file)

    # Prepare inputs and cpp flux measurements
    time = np.ascontiguousarray(cpp_solution[:, 0])
    t_exp = np.ascontiguousarray(cpp_solution[:, 1])
    flux = np.ascontiguousarray(cpp_solution[:, 2])
    flux_unc = np.ascontiguousarray(cpp_solution[:, 3])
    n_div = np.ascontiguousarray(cpp_solution[:, 4])

    # Calculate _Rust_ light curve
    lc = binary_model.compute_light_curve(time, t_exp, n_div=n_div, flux=flux)

    # Compare the Rust and C++ solution
    # Use approximate values of the difference to avoid floating point problems
    assert lc.total == pytest.approx(flux, abs=1e-100, rel=9e-9)
