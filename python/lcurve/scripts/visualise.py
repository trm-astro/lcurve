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
        fluxes = grid.flux(iangle, phase) / grid.area(iangle, phase)
        yield x, y, fluxes





def visualise():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="lcurve .mod model file to visualise")
    parser.add_argument('start_phase', help="first orbital phase to plot")
    parser.add_argument('end_phase', help="last orbital phase to plot")
    parser.add_argument('n_phases', help="Number of orbital phases to cycle through")
    parser.add_argument('--cmap', help="matplotlib colormap to plot with", default='viridis')
    args = parser.parse_args()

    start_phase = float(args.start_phase)
    end_phase = float(args.end_phase)
    n_phases = int(args.n_phases)
    cmap = args.cmap

    binary_model = lcurve.BinaryModel.from_file(args.model)

    q = binary_model.model.q.value
    iangle = binary_model.model.iangle.value

    my_cmap = copy.copy(mpl.colormaps.get_cmap(cmap))
    my_cmap.set_bad(my_cmap(0))

    fluxes1 = binary_model.star1_fine_grid.flux(iangle) / binary_model.star1_fine_grid.area(iangle)
    fluxes2 = binary_model.star2_fine_grid.flux(iangle) / binary_model.star2_fine_grid.area(iangle)
    fluxes3 = binary_model.disc_grid.flux(iangle) / binary_model.disc_grid.area(iangle)
    fluxes4 = binary_model.disc_edge_grid.flux(iangle) / binary_model.disc_edge_grid.area(iangle)
    fluxes5 = binary_model.bright_spot_grid.flux(iangle) / binary_model.bright_spot_grid.area(iangle)
    all_fluxes = np.concatenate([fluxes1, fluxes2, fluxes3, fluxes4, fluxes5])
    positive_nonzero = all_fluxes[all_fluxes > 0]

    norm_all = mpl.colors.LogNorm(positive_nonzero.min(), all_fluxes.max())

    print("Use 'Enter' key to move through phases.\n")
    fig, ax = plt.subplots()
        
    calculation1 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.star1_fine_grid, binary_model.star1_coarse_grid, True)
    calculation2 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.star2_fine_grid, binary_model.star2_coarse_grid, False)
    calculation3 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.disc_grid)
    calculation4 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.disc_edge_grid)
    calculation5 = calculation(binary_model, start_phase, end_phase, n_phases, binary_model.bright_spot_grid)

    x1, y1, fluxes1 = next(calculation1)
    x2, y2, fluxes2 = next(calculation2)
    x3, y3, fluxes3 = next(calculation3)
    x4, y4, fluxes4 = next(calculation4)
    x5, y5, fluxes5 = next(calculation5)

    star1 = ax.scatter(x1, y1, s=2, cmap=my_cmap, c=fluxes1, norm=norm_all)
    star2 = ax.scatter(x2, y2, s=2, cmap=my_cmap, c=fluxes2, norm=norm_all)
    star3 = ax.scatter(x3, y3, s=2, cmap=my_cmap, c=fluxes3, norm=norm_all)
    star4 = ax.scatter(x4, y4, s=2, cmap=my_cmap, c=fluxes4, norm=norm_all)
    star5 = ax.scatter(x5, y5, s=2, cmap=my_cmap, c=fluxes5, norm=norm_all)

    ax.set_aspect('equal')
    cofm = q / (1 + q)
    x_halfwidth = 1 + (1-roche.xl1(q)) - cofm
    y_halfwidth = x_halfwidth * np.cos(iangle * np.pi/180) + (1-roche.xl1(q))
    y_halfwidth = max(y_halfwidth, x_halfwidth*0.7)
    ax.set_ylim(-y_halfwidth, y_halfwidth)
    ax.set_xlim(-x_halfwidth, x_halfwidth)
    plt.colorbar(star1, norm=norm_all, cmap=my_cmap)

    def on_key(event):
        if event.key == "enter":
            try:
                x1, y1, fluxes1 = next(calculation1)
                x2, y2, fluxes2 = next(calculation2)
                x3, y3, fluxes3 = next(calculation3)
                x4, y4, fluxes4 = next(calculation4)
                x5, y5, fluxes5 = next(calculation5)
    
                star1.set_offsets(np.column_stack((x1, y1)))
                star1.set_array(fluxes1)
                
                star2.set_offsets(np.column_stack((x2, y2)))
                star2.set_array(fluxes2)

                star3.set_offsets(np.column_stack((x3, y3)))
                star3.set_array(fluxes3)

                star4.set_offsets(np.column_stack((x4, y4)))
                star4.set_array(fluxes4)

                star5.set_offsets(np.column_stack((x5, y5)))
                star5.set_array(fluxes5)
    
                fig.canvas.draw_idle()
    
            except StopIteration:
                ax.set_title("End phase reached!")
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()