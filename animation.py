import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


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

# prepare arrays with 2d particles
up_lim = 0.2
down_lim = -0.2

data2d = np.zeros((N, 2, frames))

for frame in range(frames):
    for i in range(N):
        particle = data[i, :, frame]

        if down_lim <= particle[2] / R0 <= up_lim:
            data2d[i, 0, frame] = particle[0]
            data2d[i, 1, frame] = particle[1]

        else:
            data2d[i, 0, frame] = 100 * R0
            data2d[i, 1, frame] = 100 * R0

print('ok')

f = 175
hist, xedges, yedges = np.histogram2d(
    data2d[:, 0, f], data[:, 1, f],
    bins=75,
    range=[[-R0, R0], [-R0, R0]]
)

fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
ax1.plot_surface(X, Y, hist.T, cmap='viridis')

plt.show()