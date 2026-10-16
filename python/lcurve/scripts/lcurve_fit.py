import numpy as np
from scipy.optimize import minimize
import argparse
import lcurve


def lcurve_fit():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="lcurve .mod model file to visualise")
    parser.add_argument('data', help="filename of data to fit")
    parser.add_argument('method', help="scipy.optimize.minimize method")
    args = parser.parse_args()

    t, te, y, ye, w, n_div = np.loadtxt(args.data, unpack=True)
    binary_model = lcurve.BinaryModel.from_file(args.model)

    t = np.ascontiguousarray(t)
    te = np.ascontiguousarray(te)
    y = np.ascontiguousarray(y)
    ye = np.ascontiguousarray(ye)
    w = np.ascontiguousarray(w)
    n_div = np.ascontiguousarray(n_div)

    variable_params = []
    starting_values = []
    for key in binary_model.model:
        if isinstance(binary_model.model[key], lcurve.Pparam):
            if binary_model.model[key]["vary"]:
                variable_params.append(key)
                starting_values.append(binary_model.model[key]["value"])
    starting_values = np.array(starting_values)

    def func_to_fit(params):
        update_dict = dict()
        for idx, par in enumerate(variable_params):
            update_dict[par] = params[idx]
        binary_model.update(update_dict)
        lc = binary_model.compute_light_curve(t, te, n_div, y, ye, w)
        return lc.chi2

    res = minimize(func_to_fit, starting_values, method=args.method)
    print(res.x)
