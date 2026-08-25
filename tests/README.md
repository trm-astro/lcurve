# Convergence tests
The testing suite here is used for testing convergence, i.e. the difference
between the legacy version of `LCURVE` written in `C++` and the `Rust`
implementation. These tests do not cover unit- or integration tests.

__Convergence__ is tested by solving the light curve in both `C++` (the
reference solution[^1]) and `Rust`, and comparing them per phase bin. Each
different "object" in the directory should ideally test different parts of the
solution pipeline.

## Adding convergence tests
If you want to add convergence tests to the testing suite, please open a pull
request (PR). The files in the PR should be structured as follows:
```text
tests
└── YOUR_OBJECT_NAME
    ├── cpp_solution.dat
    ├── description.md
    └── modfile.mod
```
If you don't have access to the `LCURVE` `C++` version, you can still open a PR
with your `modfile.mod` file. A team member will generate the `cpp_solution`
and add it to the PR.

If you generate the `cpp_solution.dat` yourself, please construct it as a
legacy `lroche` input/output file (columns: time, exposure time, n_div, flux,
flux uncertainty, weights1, weights2).

The `description.md` should include a short description of the object, e.g.
Period, Types of components, and specific physics included in the test (e.g.
reflection effect, ellipsoidal modulation, eclipse).


[^1]: As of 2026-08-20 the `C++` version of `LCURVE` is considered the ground
truth to which we compare. Due to known bugs in `LCURVE`, the `Rust`
version will diverge again over time, after we can be sure that it does not
have any severe differences in the calculated physics.
