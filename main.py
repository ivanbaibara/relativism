import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numba as nb
import time
from analytic_grafs import Rmy, Rho, rho
from constants import *

sigma = 0.2
mu = 0
tau = 3

def fill_particles_const():
    global N
    global particle_r
    global r_max

    r_max2 = r_max ** 2

    count = 0
    while count < N:
        x = 2 * np.random.rand() * r_max - r_max
        y = 2 * np.random.rand() * r_max - r_max
        z = 2 * np.random.rand() * r_max - r_max

        r2 = (x ** 2 + y ** 2 + z ** 2)
        if r2 < r_max2:
            particle_r[count, 0] = x
            particle_r[count, 1] = y
            particle_r[count, 2] = z

            count += 1

@nb.njit()
def fill_lognorm_spherical(particle_r, tau=tau, mu=mu, sigma=sigma, max_radius_lognorm=1):
    global N, r_max

    lognorm = lambda r: 1 / (sigma * r ** 3) * np.exp(- 1 / (2 * sigma ** 2) * (np.log(tau * r) - mu) ** 2)
    lognorm_max = lognorm(np.exp(mu - 3 * sigma ** 2) / tau)

    lognorm_prob = lambda r: lognorm(r) / lognorm_max

    access_p = lambda r: True if np.random.rand() < lognorm_prob(r) else False

    count = 0
    while count < N:
        r = max_radius_lognorm * np.random.rand() ** (1/3)

        if access_p(r):
            r1 = r * r_max / max_radius_lognorm
            phi = 2 * np.random.rand() * np.pi
            theta = np.arccos(2 * np.random.rand() - 1)

            x1 = r1 * np.sin(theta) * np.cos(phi)
            y1 = r1 * np.sin(theta) * np.sin(phi)
            z1 = r1 * np.cos(theta)

            particle_r[count, :] = [x1, y1, z1]
            count += 1

@nb.njit(parallel=True,cache=True)
def calc_forces(particle_r, particle_f, k, q, N):
    eps = 1e-90

    # проход по всем частицам
    for i in nb.prange(N):
        for j in range(i + 1, N):
            dx = particle_r[i, 0] - particle_r[j, 0]
            dy = particle_r[i, 1] - particle_r[j, 1]
            dz = particle_r[i, 2] - particle_r[j, 2]

            l2 = dx * dx + dy * dy + dz * dz

            # избегаем деления на 0
            if l2 < eps:
                l2 = eps

            factor = k * q * q / (l2 * np.sqrt(l2))

            f_x = factor * dx
            f_y = factor * dy
            f_z = factor * dz

            particle_f[i, 0] += f_x
            particle_f[i, 1] += f_y
            particle_f[i, 2] += f_z

            particle_f[j, 0] -= f_x
            particle_f[j, 1] -= f_y
            particle_f[j, 2] -= f_z

def calc_new_pos6_vv():
    global particle_r
    global particle_v
    global particle_v_half
    global particle_f
    global particle_a
    global m, N, q, total_q

    # вычисляем ускорение
    particle_a = particle_f / m

    # обновляем положение и вычисляем скорость на пол шаге
    particle_r += particle_v * dt + 0.5 * particle_a * dt * dt
    particle_v_half = particle_v + 0.5 * particle_a * dt

    # вычисляем новую силу (ускорение) на основе новых координат
    particle_f.fill(0)

    calc_forces(
        particle_r,
        particle_f,
        k,
        q,
        N
    )

    particle_a = particle_f / m

    # завершаем обновление скорости
    particle_v = particle_v_half + 0.5 * particle_a * dt

def calc_new_pos6_vv_relativism():
    global particle_r
    global particle_v
    global particle_v_half
    global particle_f
    global particle_a
    global m, N, c

    # вычисляем ускорение (релятивизм)
    particle_v2 = particle_v[:, 0] ** 2 + particle_v[:, 1] ** 2 + particle_v[:, 2] ** 2
    gamma = 1 / np.sqrt(1 - particle_v2 / (c ** 2))

    particle_a[:, 0] = 1 / (m * gamma[:]) * (
        (1 - particle_v[:, 0] ** 2 / c ** 2) * particle_f[:, 0] -
        (particle_v[:, 0] * particle_v[:, 1] / c ** 2) * particle_f[:, 1] -
        (particle_v[:, 0] * particle_v[:, 2] / c ** 2) * particle_f[:, 2]
    )
    particle_a[:, 1] = 1 / (m * gamma[:]) * (
            -(particle_v[:, 0] * particle_v[:, 1] / c ** 2) * particle_f[:, 0] +
            (1 - particle_v[:, 1] ** 2 / c ** 2) * particle_f[:, 1] -
            (particle_v[:, 1] * particle_v[:, 2] / c ** 2) * particle_f[:, 2]
    )
    particle_a[:, 2] = 1 / (m * gamma[:]) * (
            -(particle_v[:, 0] * particle_v[:, 2] / c ** 2) * particle_f[:, 0] -
            (particle_v[:, 1] * particle_v[:, 2] / c ** 2) * particle_f[:, 1] +
            (1 - particle_v[:, 2] ** 2 / c ** 2) * particle_f[:, 2]
    )

    # обновляем положение и вычисляем скорость на пол шаге
    particle_r += particle_v * dt + 0.5 * particle_a * dt * dt
    particle_v_half = particle_v + 0.5 * particle_a * dt

    # вычисляем новую силу (ускорение) на основе новых координат
    particle_f.fill(0)
    particle_a.fill(0)

    calc_forces(
        particle_r,
        particle_f,
        k,
        q,
        N
    )

    # опять вычисляем ускорение
    particle_v2 = particle_v[:, 0] ** 2 + particle_v[:, 1] ** 2 + particle_v[:, 2] ** 2
    gamma = 1 / np.sqrt(1 - particle_v2 / (c ** 2))

    particle_a[:, 0] = 1 / (m * gamma[:]) * (
            (1 - particle_v[:, 0] ** 2 / c ** 2) * particle_f[:, 0] -
            (particle_v[:, 0] * particle_v[:, 1] / c ** 2) * particle_f[:, 1] -
            (particle_v[:, 0] * particle_v[:, 2] / c ** 2) * particle_f[:, 2]
    )
    particle_a[:, 1] = 1 / (m * gamma[:]) * (
            -(particle_v[:, 0] * particle_v[:, 1] / c ** 2) * particle_f[:, 0] +
            (1 - particle_v[:, 1] ** 2 / c ** 2) * particle_f[:, 1] -
            (particle_v[:, 1] * particle_v[:, 2] / c ** 2) * particle_f[:, 2]
    )
    particle_a[:, 2] = 1 / (m * gamma[:]) * (
            -(particle_v[:, 0] * particle_v[:, 2] / c ** 2) * particle_f[:, 0] -
            (particle_v[:, 1] * particle_v[:, 2] / c ** 2) * particle_f[:, 1] +
            (1 - particle_v[:, 2] ** 2 / c ** 2) * particle_f[:, 2]
    )

    # завершаем обновление скорости
    particle_v = particle_v_half + 0.5 * particle_a * dt


# global vars
N = 55000
total_q = Q0
total_m = M0 # протоны
q = total_q / N
m = total_m / N
r_max = R0
t0_max = T0
dt = t0_max * 0.01 / 4

shots = 45 * 4

particle_r = np.zeros((N, 3))
particle_v = np.zeros((N, 3))
particle_v_half = np.zeros((N, 3))
particle_f = np.zeros((N, 3))
particle_a = np.zeros((N, 3))
fill_lognorm_spherical(particle_r)
particle_r_init = particle_r.copy()

# for .npz file
info_keys = np.array(['N', 'M0', 'R0', 'T0', 'G', 'c', 'dt', 'frames', 'sigma', 'mu', 'tau'])
info_values = np.array([N, M0, R0, T0, k, c, dt, shots, sigma, mu, tau])

data = np.zeros((N, 3, shots))

calc_forces(
        particle_r,
        particle_f,
        k,
        q,
        N
    )

for i in range(shots):
    start_time = time.time()
    calc_new_pos6_vv()
    # save to arr
    data[:, :, i] = particle_r[:, :]
    print(f'{i} with time: {time.time() - start_time} s')


tm = time.localtime()

save_name = f'{tm.tm_mday}-{tm.tm_mon}-{tm.tm_year}_{tm.tm_hour}-{tm.tm_min}_el_{N}'
print(save_name)

np.savez(f'solved/{save_name}',
         info_keys=info_keys,
         info_values=info_values,
         data=data
         )


def compare_density_plots(particle_r_initial, particle_r_final,
                                   percentile_cut=100, bins=150,
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

    # Плотности заряда
    densities_initial = counts_initial / shell_volumes_initial * q
    densities_final = counts_final / shell_volumes_final * q

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
            label=f'Конец (t={(dt * shots)}c)')

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
    ax1.set_ylabel('Плотность заряда, Кл/м³')  # Изменено название оси
    ax1.set_title(f'Радиальные плотности заряда (отсечение {percentile_cut}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if x_lim and y_lim:
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)

    plt.show()


N_points1 = 1000
xn1 = np.array([R0 * (0.1 + i / N_points1) for i in range(N_points1)])
yn1 = np.array([rho(x) for x in xn1])

N_points2 = 100
x_0 = np.array([R0 * (0.1 + i / N_points2) for i in range(N_points2)])
t1 = dt * shots
xn2 = np.array([Rmy(x, t1) for x in x_0])
yn2 = np.array([Rho(x, t1) for x in x_0])


compare_density_plots(
    particle_r_init,
    particle_r,
    xn1=xn1,
    yn1=yn1,
    xn2=xn2,
    yn2=yn2,
    x_lim=[0, R0 * 1.3],
    y_lim=[0, 1e27],
    analytical_marker='',
    analytical_color1='grey',
    analytical_color2='cyan',
    analytical_size=2
)
