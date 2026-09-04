import numpy as np
import matplotlib.pyplot as plt
import lcurve

from importlib.resources import files, as_file



def example():
    resource = files("lcurve").joinpath("examples", "example_model.mod")

    with as_file(resource) as model_path:
        binary_model = lcurve.BinaryModel.from_file(str(model_path))

    times = np.linspace(0.035, 0.0525, 1000)
    exposure_times = 1.0e-5 * np.ones_like(times)
    ndiv = np.ones_like(times)

    lc = binary_model.compute_light_curve(times, exposure_times, ndiv)

    fig, ax = plt.subplots()
    ax.plot(times, lc.total)
    plt.show()