import numpy as np
import matplotlib.pyplot as plt


def density_plot(
        particle_r,
        percentile_cut=100,
        bins=100,
        x_lim=None,
        y_lim=None
):
    r_vals = np.sqrt(np.sum(particle_r ** 2, axis=1))

    cutoff = np.percentile(r_vals, percentile_cut)
    mask = r_vals <= cutoff

    r_filtered = r_vals[mask]

    bin_edges = np.linspace(0, cutoff, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    counts, _ = np.histogram(r_filtered, bins=bin_edges)

    shell_volumes = 4 * np.pi * bin_centers ** 2 * bin_widths
    if shell_volumes[0] == 0 and len(shell_volumes) > 1:
        shell_volumes[0] = shell_volumes[1]

    densities = counts / shell_volumes

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.bar(bin_centers, densities,
            width=bin_widths * 0.9,
            alpha=0.7, color='blue', edgecolor='black',
            label=f'Начало')

    ax.set_xlabel('Радиус')
    ax.set_ylabel('Плотность')
    ax.set_title(f'Радиальные плотности заряда (отсечение {percentile_cut})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()


def compare_density_plots(particle_r_initial, particle_r_final, q,
                                   percentile_cut=100, bins=150,
                                   xn1=None, yn1=None, xn2=None, yn2=None,
                                   analytical_color1='green', analytical_color2='purple',
                                   analytical_label1='Аналит. начало', analytical_label2='Аналит. конец',
                                   analytical_marker='o', analytical_size=30,
                                   x_lim=None, y_lim=None):
    r_vals_initial = np.sqrt(np.sum(particle_r_initial ** 2, axis=1))
    r_vals_final = np.sqrt(np.sum(particle_r_final ** 2, axis=1))

    cutoff_initial = np.percentile(r_vals_initial, percentile_cut)
    cutoff_final = np.percentile(r_vals_final, percentile_cut)

    mask_initial = r_vals_initial <= cutoff_initial
    mask_final = r_vals_final <= cutoff_final

    r_filtered_initial = r_vals_initial[mask_initial]
    r_filtered_final = r_vals_final[mask_final]

    bin_edges_initial = np.linspace(0, cutoff_initial, bins + 1)
    bin_edges_final = np.linspace(0, cutoff_final, bins + 1)

    bin_centers_initial = (bin_edges_initial[:-1] + bin_edges_initial[1:]) / 2
    bin_centers_final = (bin_edges_final[:-1] + bin_edges_final[1:]) / 2

    bin_widths_initial = bin_edges_initial[1:] - bin_edges_initial[:-1]
    bin_widths_final = bin_edges_final[1:] - bin_edges_final[:-1]

    counts_initial, _ = np.histogram(r_filtered_initial, bins=bin_edges_initial)
    counts_final, _ = np.histogram(r_filtered_final, bins=bin_edges_final)

    shell_volumes_initial = 4 * np.pi * bin_centers_initial ** 2 * bin_widths_initial
    shell_volumes_final = 4 * np.pi * bin_centers_final ** 2 * bin_widths_final

    if shell_volumes_initial[0] == 0 and len(shell_volumes_initial) > 1:
        shell_volumes_initial[0] = shell_volumes_initial[1]
    if shell_volumes_final[0] == 0 and len(shell_volumes_final) > 1:
        shell_volumes_final[0] = shell_volumes_final[1]

    densities_initial = counts_initial / shell_volumes_initial * q
    densities_final = counts_final / shell_volumes_final * q

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    ax1.bar(bin_centers_initial, densities_initial,
            width=bin_widths_initial * 0.9,
            alpha=0.7, color='blue', edgecolor='black',
            label=f'Начало')

    ax1.bar(bin_centers_final, densities_final,
            width=bin_widths_final * 0.9,
            alpha=0.7, color='red', edgecolor='black',
            label=f'Конец')

    if xn1 is not None and yn1 is not None:
        ax1.plot(xn1, yn1, marker=analytical_marker,
                    color=analytical_color1, linewidth=analytical_size,
                    label=analytical_label1)

    if xn2 is not None and yn2 is not None:
        ax1.plot(xn2, yn2, marker=analytical_marker,
                    color=analytical_color2, linewidth=analytical_size,
                    label=analytical_label2)

    ax1.set_xlabel('Радиус, r')
    ax1.set_ylabel('Плотность заряда')
    ax1.set_title(f'Радиальные плотности заряда (отсечение {percentile_cut}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if x_lim and y_lim:
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)

    plt.show()