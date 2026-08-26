#!/usr/bin/env python3

usage = """ Plots the flux calculated by lcurve from the specified .mod
file. The plot has the total flux and the flux of the individual 
components within the system.

The default number of data points is given by range(0,period,0.01) where period
is the period in the .mod file. So if this is 1 you get 100 data points.
You can change this by using the argument t_exp."""

import argparse
import lroche
import numpy as np
import matplotlib.pyplot as plt

def plot_model(t,lc,ax) -> None:

    if all(lc.star1) != 0:
        ax.plot(t, lc.star1, label='Star 1')
    if all(lc.star2) != 0:
        ax.plot(t, lc.star2, label='Star 2')
    if all(lc.disc) != 0:
        ax.plot(t, lc.disc, label='Disc')
    if all(lc.disc_edge) != 0:
        ax.plot(t, lc.disc_edge, label='Disc edge')
    if all(lc.bright_spot) != 0:
        ax.plot(t, lc.bright_spot, label='Bright spot')
    ax.plot(t, lc.total, label='Total')
    ax.legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "model",
        help=".mod file specifying the model to be computed by lcurve.",
    )

    parser.add_argument(
        "-dt",
        dest="t_exp",
        type=float,
        default=0.01,
        help="Time spacing between computed fluxes."
    )
    
    args = parser.parse_args()

    binary_model = lroche.BinaryModel.from_file(args.model)
    period = binary_model.model.period.value
    t_exp = args.t_exp

    t = np.arange(0,period,t_exp)
    t_exp = np.full_like(t,fill_value=t_exp)
    t = np.ascontiguousarray(t)
    t_exp = np.ascontiguousarray(t_exp)

    lc = binary_model.compute_light_curve(t,t_exp)

    fig,ax = plt.subplots()
    plot_model(t,lc,ax)
    plt.show()
