import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import copy
import roche
import lcurve


def calculation(binary_model, start, end, n, grid_fine, grid_coarse=None, star1=False):
    q = binary_model.model.q.value
    iangle = binary_model.model.iangle.value
    wavelength = binary_model.model.wavelength

    phase1 = binary_model.model.phase1
    phase2 = binary_model.model.phase2

    for phase in np.linspace(start, end, n):
        phase %= 1
        if phase > 0.5:
            phase -= 1
        if (abs(phase) > phase1) & (grid_coarse is not None) & star1:
            grid = grid_coarse
        elif ((abs(phase) < phase2) | (abs(phase) > 1-phase2)) & (grid_coarse is not None) & ~star1:
            grid = grid_coarse
        else:
            grid = grid_fine
        
        x, y = grid.project_2d(q, iangle, phase)
        values = grid.temperature(wavelength, iangle, phase)
        yield x, y, values





def visualise():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="lcurve .mod model file to visualise")
    parser.add_argument('start_phase', help="first orbital phase to plot")
    parser.add_argument('end_phase', help="last orbital phase to plot")
    parser.add_argument('n_phases', help="Number of orbital phases to cycle through")
    parser.add_argument('--cmap', help="matplotlib colormap to plot with", default='viridis')
    parser.add_argument('--cmap1', help="matplotlib colormap to plot with", default=None)
    parser.add_argument('--cmap2', help="matplotlib colormap to plot with", default=None)
    parser.add_argument('--normalisation', help="matplotlib colourmap normalisation", default='Norm')
    parser.add_argument('--background', help="plot background color", default='w')
    args = parser.parse_args()

    start_phase = float(args.start_phase)
    end_phase = float(args.end_phase)
    n_phases = int(args.n_phases)
    cmap = args.cmap
    cmap1 = args.cmap1
    cmap2 = args.cmap2

    normalisation_dict = dict(
        Norm = mpl.colors.Normalize,
        LogNorm = mpl.colors.LogNorm,
    )
    norm_func = normalisation_dict[args.normalisation]

    binary_model = lcurve.BinaryModel.from_file(args.model)

    q = binary_model.model.q.value
    iangle = binary_model.model.iangle.value
    wavelength = binary_model.model.wavelength

    my_cmap = copy.copy(mpl.colormaps.get_cmap(cmap))
    my_cmap.set_bad(my_cmap(0))

    values1 = binary_model.star1_fine_grid.temperature(wavelength, iangle)
    values2 = binary_model.star2_fine_grid.temperature(wavelength, iangle)
    values3 = binary_model.disc_grid.temperature(wavelength, iangle)
    values4 = binary_model.disc_edge_grid.temperature(wavelength, iangle)
    values5 = binary_model.bright_spot_grid.temperature(wavelength, iangle)
    all_fluxes = np.concatenate([values1, values2, values3, values4, values5])
    positive_nonzero = all_fluxes[all_fluxes > 0]

    
    norm_all = norm_func(positive_nonzero.min(), all_fluxes.max())

    if args.cmap1 and args.cmap2:
        values_minus_star2 = np.concatenate([values1, values3, values4, values5])
        norm1 = norm_func(vmin=values_minus_star2.min(), vmax=values_minus_star2.max())
        norm2 = norm_func(vmin=values2.min(), vmax=values2.max())
        my_cmap1 = copy.copy(mpl.colormaps.get_cmap(cmap1))
        my_cmap1.set_bad(my_cmap1(0))
        my_cmap2 = copy.copy(mpl.colormaps.get_cmap(cmap2))
        my_cmap2.set_bad(my_cmap1(0))

    else:
        norm1 = norm_all
        norm2 = norm_all
        my_cmap1 = my_cmap
        my_cmap2 = my_cmap

    print("Use 'Enter' key to move through phases.\n")
    fig, ax = plt.subplots()

    ax.set_facecolor(args.background)
    calculation1 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.star1_fine_grid, binary_model.star1_coarse_grid, True)
    calculation2 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.star2_fine_grid, binary_model.star2_coarse_grid, False)
    calculation3 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.disc_grid)
    calculation4 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.disc_edge_grid)
    calculation5 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.bright_spot_grid)

    x1, y1, values1 = next(calculation1)
    x2, y2, values2 = next(calculation2)
    x3, y3, values3 = next(calculation3)
    x4, y4, values4 = next(calculation4)
    x5, y5, values5 = next(calculation5)

    star1 = ax.scatter(x1, y1, s=2, cmap=my_cmap1, c=values1, norm=norm1)
    star2 = ax.scatter(x2, y2, s=2, cmap=my_cmap2, c=values2, norm=norm2)
    star3 = ax.scatter(x3, y3, s=2, cmap=my_cmap1, c=values3, norm=norm1)
    star4 = ax.scatter(x4, y4, s=2, cmap=my_cmap1, c=values4, norm=norm1)
    star5 = ax.scatter(x5, y5, s=2, cmap=my_cmap1, c=values5, norm=norm1)

    ax.set_aspect('equal')
    cofm = q / (1 + q)
    x_halfwidth = 1 + (1-roche.xl1(q)) - cofm
    y_halfwidth = x_halfwidth * np.cos(iangle * np.pi/180) + (1-roche.xl1(q))
    y_halfwidth = max(y_halfwidth, x_halfwidth*0.7)
    ax.set_ylim(-y_halfwidth, y_halfwidth)
    ax.set_xlim(-x_halfwidth, x_halfwidth)
    cbar1 = plt.colorbar(star1, norm=norm1, cmap=my_cmap1, shrink=0.7)

    if args.cmap1 and args.cmap2:
        cbar2 = plt.colorbar(star2, norm=norm2, cmap=my_cmap2)
        cbar1.set_label("star1")
        cbar2.set_label("star2")

    def on_key(event):
        if event.key == "enter":
            try:
                x1, y1, values1 = next(calculation1)
                x2, y2, values2 = next(calculation2)
                x3, y3, values3 = next(calculation3)
                x4, y4, values4 = next(calculation4)
                x5, y5, values5 = next(calculation5)
    
                star1.set_offsets(np.column_stack((x1, y1)))
                star1.set_array(values1)
                
                star2.set_offsets(np.column_stack((x2, y2)))
                star2.set_array(values2)

                star3.set_offsets(np.column_stack((x3, y3)))
                star3.set_array(values3)

                star4.set_offsets(np.column_stack((x4, y4)))
                star4.set_array(values4)

                star5.set_offsets(np.column_stack((x5, y5)))
                star5.set_array(values5)
    
                fig.canvas.draw_idle()
    
            except StopIteration:
                ax.set_title("End phase reached!")
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()