import numpy as np
import matplotlib.pyplot as plt
from src.utils import compare_density_plots

filename = 'solved/14-3-2026_22-6_el_25000.npz'

solved_info = np.load(
    filename
)

# ['N', 'q', 'm', 'R0', 'T0', 'G', 'c', 'dt', 'frames', 'sigma', 'mu', 'tau']

info_keys = solved_info['info_keys']
info_values = solved_info['info_values']
data = solved_info['data']

info = dict(zip(info_keys, info_values))

N = int(info['N'])
Q0 = float(info['q'])
M0 = float(info['m'])
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
plt.bar([x for x in range(layers_count)], particles_in_layer)
plt.show()

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


