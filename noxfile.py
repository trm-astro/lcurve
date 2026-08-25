import nox


@nox.session(python=["3.11", "3.12"])
def tests(session):
    # Load pyproject.toml
    pyproject = nox.project.load_toml("pyproject.toml")
    # Install dependencies
    session.install(*nox.project.dependency_groups(pyproject, "tests"))
    # Run tests
    session.run("pytest")
