from analytic_solution import AnalyticSolutionGV
import numpy as np
import matplotlib.pyplot as plt

def compare_density_plots(particle_r_initial, particle_r_final,
                                   percentile_cut=98, bins=110,
                                   xn1=None, yn1=None, xn2=None, yn2=None,
                                   analytical_color1='green', analytical_color2='purple',
                                   analytical_label1='Аналит. начало', analytical_label2='Аналит. конец',
                                   analytical_marker='o', analytical_size=30,
                                   x_lim=None, y_lim=None):
    # Вычисляем радиусы
    r_vals_initial = np.sqrt(np.sum(particle_r_initial ** 2, axis=1))
    r_vals_final = np.sqrt(np.sum(particle_r_final ** 2, axis=1))

    # Разные радиусы отсечения для начального и конечного состояний
    cutoff_initial = np.percentile(r_vals_initial, percentile_cut)
    cutoff_final = np.percentile(r_vals_final, percentile_cut)

    # Отсекаем частицы своими порогами
    mask_initial = r_vals_initial <= cutoff_initial
    mask_final = r_vals_final <= cutoff_final

    r_filtered_initial = r_vals_initial[mask_initial]
    r_filtered_final = r_vals_final[mask_final]

    # Собственные столбцы для каждого распределения
    bin_edges_initial = np.linspace(0, cutoff_initial, bins + 1)
    bin_edges_final = np.linspace(0, cutoff_final, bins + 1)

    bin_centers_initial = (bin_edges_initial[:-1] + bin_edges_initial[1:]) / 2
    bin_centers_final = (bin_edges_final[:-1] + bin_edges_final[1:]) / 2

    bin_widths_initial = bin_edges_initial[1:] - bin_edges_initial[:-1]
    bin_widths_final = bin_edges_final[1:] - bin_edges_final[:-1]

    # Гистограммы
    counts_initial, _ = np.histogram(r_filtered_initial, bins=bin_edges_initial)
    counts_final, _ = np.histogram(r_filtered_final, bins=bin_edges_final)

    # Объемы оболочек
    shell_volumes_initial = 4 * np.pi * bin_centers_initial ** 2 * bin_widths_initial
    shell_volumes_final = 4 * np.pi * bin_centers_final ** 2 * bin_widths_final

    if shell_volumes_initial[0] == 0 and len(shell_volumes_initial) > 1:
        shell_volumes_initial[0] = shell_volumes_initial[1]
    if shell_volumes_final[0] == 0 and len(shell_volumes_final) > 1:
        shell_volumes_final[0] = shell_volumes_final[1]

    # Плотности массы
    densities_initial = counts_initial / shell_volumes_initial * M0 / N
    densities_final = counts_final / shell_volumes_final * M0 / N

    # График с двумя осями X
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Панель 1: ОТДЕЛЬНЫЕ гистограммы
    ax1.bar(bin_centers_initial, densities_initial,
            width=bin_widths_initial * 0.9,
            alpha=0.7, color='blue', edgecolor='black',
            label=f'Начало (t={0}c)')

    ax1.bar(bin_centers_final, densities_final,
            width=bin_widths_final * 0.9,
            alpha=0.7, color='red', edgecolor='black',
            label=f'Конец (t={(dt * frames)}c)')

    # Наложение аналитических кривых точками, если предоставлены
    if xn1 is not None and yn1 is not None:
        ax1.plot(xn1, yn1, marker=analytical_marker,
                    color=analytical_color1, linewidth=analytical_size,
                    label=analytical_label1)

    if xn2 is not None and yn2 is not None:
        ax1.plot(xn2, yn2, marker=analytical_marker,
                    color=analytical_color2, linewidth=analytical_size,
                    label=analytical_label2)

    ax1.set_xlabel('Радиус, r')
    ax1.set_ylabel('Плотность массы, кг/м³')  # Изменено название оси
    ax1.set_title(f'Радиальные плотности массы (отсечение {percentile_cut}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if x_lim and y_lim:
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)

    plt.show()

filename = 'solved/20-2-2026_19-15_el_55000.npz'

solved_info = np.load(
    filename
)

# ['N', 'M0', 'R0', 'dt', 'frames', 'sigma', 'm', 'tau']

info_keys = solved_info['info_keys']
info_values = solved_info['info_values']
data = solved_info['data']

info = dict(zip(info_keys, info_values))

N = int(info['N'])
M0 = float(info['M0'])
R0 = float(info['R0'])
T0 = float(info['T0'])
frames = int(info['frames'])
dt = float(info['dt'])

sigma = float(info['sigma'])
mu = float(info['mu'])
tau = 3 / R0

G = float(info['G'])
c = float(info['c'])

'''
asGV = AnalyticSolutionGV(
    sigma, mu, tau, M0, R0, G, c
)

frame = 100
t0 = dt * frame
N_frame = 1000



xs = np.array([R0 * (0.05 + i / N_frame) for i in range(N_frame)])
y0 = np.array([asGV.rho(x) for x in xs])

xn = np.array([asGV.Rmy(t0, x) for x in xs])
yn_r = np.array([asGV.Rho(x, t0) for x in xs])


compare_density_plots(
    data[:, :, 0],
    data[:, :, int(frame) - 1],
    xn1=xs, yn1=y0,
    xn2=xn, yn2=yn_r,
    analytical_marker='',
    analytical_size=2
)
'''

data_r = np.zeros((N, frames))

# prepare data
data_r[:, :] = (data[:, 0, :] ** 2 + data[:, 1, :] ** 2 + data[:, 2, :] ** 2) ** 0.5

# step 1 (find all fly-out particles)
flyout_count = 0

for i in range(N):
    if data_r[i, frames - 1] > R0:
        flyout_count += 1
        data_r[i,:] = 0

print(f'Количество вылетевших частиц из системы: {flyout_count}')

# step 2 (form indexes array for layers)
layers_count = 100
particle_in_layer_index = [[] for _ in range(layers_count)]

dr = R0 / layers_count

for i in range(N):
    if 0 < data_r[i, 0] < R0 - dr:
        layer_index = int(data_r[i, 0] / dr)
        particle_in_layer_index[layer_index].append(i)

particles_in_layer = [len(x) for x in particle_in_layer_index]

# show particles count by layers
#plt.bar([x for x in range(layers_count)], particles_in_layer)
#plt.show()

# step 3 (make lines for average particle r in layer
layers_trace = [[] for _ in range(layers_count)]

for frame_index in range(frames):
    # calc mean r value by indexes in layers
    for layer_index in range(layers_count):
        # skip empty levels
        if particles_in_layer[layer_index] == 0:
            continue

        mean = 0
        for particle_index in particle_in_layer_index[layer_index]:
            mean += data_r[particle_index, frame_index]

        mean /= particles_in_layer[layer_index]

        layers_trace[layer_index].append(mean)

# step 5 (count non-empty layers)
non_empty_layers = []

for i in range(layers_count):
    if particles_in_layer[i] != 0:
        non_empty_layers.append(i)

# step 6 (show)
print('last prepare')
for l in non_empty_layers:
    plt.plot(layers_trace[l], [x for x in range(frames)])


plt.xlabel('Радиус, r')
plt.ylabel('Итерации')
plt.show()


