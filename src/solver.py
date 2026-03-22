import numpy as np
import numba as nb
import time


class Solver:
    def __init__(self):
        self.mu = None
        self.tau = None
        self.sigma = None

        self.c = None
        self.k = None
        self.N = None
        self.q = None
        self.m = None
        self.r_max = None
        self.t_max = None
        self.dt = None
        self.frames = None

        self.particle_r = None
        self.particle_v = None
        self.particle_v_half = None
        self.particle_a = None
        self.particle_f = None

        self.data = None

    def set_constants(self, N, charge, mass, k, c, r_max, t_max, dt, sigma=0.2, tau=3, mu=0):
        self.N = N
        self.q = charge / self.N
        self.m = mass / self.N
        self.k = k
        self.c = c
        self.r_max = r_max
        self.t_max = t_max
        self.dt = dt
        self.frames = int(self.t_max / self.dt)

        self.tau = tau
        self.mu = mu
        self.sigma = sigma

        shape = (self.N, 3)
        self.particle_r = np.zeros(shape)
        self.particle_v = np.zeros(shape)
        self.particle_v_half = np.zeros(shape)
        self.particle_a = np.zeros(shape)
        self.particle_f = np.zeros(shape)

        self.data = np.zeros((self.N, 3, self.frames))

    @staticmethod
    @nb.njit()
    def fill_lognorm_spherical(particle_r, r_max, N, tau, mu, sigma):
        lognorm = lambda r: 1 / (sigma * r ** 3) * np.exp(- 1 / (2 * sigma ** 2) * (np.log(tau * r) - mu) ** 2)
        lognorm_max = lognorm(np.exp(mu - 3 * sigma ** 2) / tau)

        lognorm_prob = lambda r: lognorm(r) / lognorm_max

        access_p = lambda r: True if np.random.rand() < lognorm_prob(r) else False

        count = 0
        while count < N:
            r = np.random.rand() ** (1 / 3)

            if access_p(r):
                r1 = r * r_max
                phi = 2 * np.random.rand() * np.pi
                theta = np.arccos(2 * np.random.rand() - 1)

                x1 = r1 * np.sin(theta) * np.cos(phi)
                y1 = r1 * np.sin(theta) * np.sin(phi)
                z1 = r1 * np.cos(theta)

                particle_r[count, :] = [x1, y1, z1]
                count += 1

    def fill_particles(self, particle_r=None):
        if particle_r is None:
            Solver.fill_lognorm_spherical(
                self.particle_r,
                self.r_max,
                self.N,
                self.tau,
                self.mu,
                self.sigma
            )

        else:
            self.particle_r = particle_r

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def calc_forces(particle_r, particle_f, k, q, N):
        eps = 1e-90

        for i in nb.prange(N):
            for j in range(i + 1, N):
                dx = particle_r[i, 0] - particle_r[j, 0]
                dy = particle_r[i, 1] - particle_r[j, 1]
                dz = particle_r[i, 2] - particle_r[j, 2]

                l2 = dx * dx + dy * dy + dz * dz
                l2 = max(l2, eps)

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

    def calc_new_pos6_vv(self):
        self.particle_a = self.particle_f / self.m

        self.particle_r += self.particle_v * self.dt + 0.5 * self.particle_a * self.dt * self.dt
        self.particle_v_half = self.particle_v + 0.5 * self.particle_a * self.dt

        self.particle_f.fill(0)

        Solver.calc_forces(
            self.particle_r,
            self.particle_f,
            self.k,
            self.q,
            self.N
        )

        self.particle_a = self.particle_f / self.m
        self.particle_v = self.particle_v_half + 0.5 * self.particle_a * self.dt

    def calc_new_pos6_v_relativism(self):
        particle_v2 = self.particle_v[:, 0] ** 2 + self.particle_v[:, 1] ** 2 + self.particle_v[:, 2] ** 2
        gamma = 1 / np.sqrt(1 - particle_v2 / (self.c ** 2))

        self.particle_a[:, 0] = 1 / (self.m * gamma[:]) * (
                (1 - self.particle_v[:, 0] ** 2 / self.c ** 2) * self.particle_f[:, 0] -
                (self.particle_v[:, 0] * self.particle_v[:, 1] / self.c ** 2) * self.particle_f[:, 1] -
                (self.particle_v[:, 0] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 2]
        )
        self.particle_a[:, 1] = 1 / (self.m * gamma[:]) * (
                -(self.particle_v[:, 0] * self.particle_v[:, 1] / self.c ** 2) * self.particle_f[:, 0] +
                (1 - self.particle_v[:, 1] ** 2 / self.c ** 2) * self.particle_f[:, 1] -
                (self.particle_v[:, 1] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 2]
        )
        self.particle_a[:, 2] = 1 / (self.m * gamma[:]) * (
                -(self.particle_v[:, 0] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 0] -
                (self.particle_v[:, 1] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 1] +
                (1 - self.particle_v[:, 2] ** 2 / self.c ** 2) * self.particle_f[:, 2]
        )

        self.particle_r += self.particle_v * self.dt + 0.5 * self.particle_a * self.dt * self.dt
        particle_v_half = self.particle_v + 0.5 * self.particle_a * self.dt

        self.particle_f.fill(0)
        self.particle_a.fill(0)

        Solver.calc_forces(
            self.particle_r,
            self.particle_f,
            self.k,
            self.q,
            self.N
        )

        particle_v2 = self.particle_v[:, 0] ** 2 + self.particle_v[:, 1] ** 2 + self.particle_v[:, 2] ** 2
        gamma = 1 / np.sqrt(1 - particle_v2 / (self.c ** 2))

        self.particle_a[:, 0] = 1 / (self.m * gamma[:]) * (
                (1 - self.particle_v[:, 0] ** 2 / self.c ** 2) * self.particle_f[:, 0] -
                (self.particle_v[:, 0] * self.particle_v[:, 1] / self.c ** 2) * self.particle_f[:, 1] -
                (self.particle_v[:, 0] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 2]
        )
        self.particle_a[:, 1] = 1 / (self.m * gamma[:]) * (
                -(self.particle_v[:, 0] * self.particle_v[:, 1] / self.c ** 2) * self.particle_f[:, 0] +
                (1 - self.particle_v[:, 1] ** 2 / self.c ** 2) * self.particle_f[:, 1] -
                (self.particle_v[:, 1] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 2]
        )
        self.particle_a[:, 2] = 1 / (self.m * gamma[:]) * (
                -(self.particle_v[:, 0] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 0] -
                (self.particle_v[:, 1] * self.particle_v[:, 2] / self.c ** 2) * self.particle_f[:, 1] +
                (1 - self.particle_v[:, 2] ** 2 / self.c ** 2) * self.particle_f[:, 2]
        )

        self.particle_v = particle_v_half + 0.5 * self.particle_a * self.dt

    def run(self):
        print(f'Total frames: {self.frames}')
        for i in range(self.frames):
            start_time = time.time()
            self.calc_new_pos6_v_relativism()
            self.data[:, :, i] = self.particle_r[:, :]

            print(f'frame {i} with time: {time.time() - start_time} s')

    def final(self):
        return self.particle_r

    def save_data(self, directory):
        tm = time.localtime()
        filename = f'{tm.tm_mday}-{tm.tm_mon}-{tm.tm_year}_{tm.tm_hour}-{tm.tm_min}_el_{self.N}'

        info_keys = np.array(['N', 'q', 'm', 'R0', 'T0', 'G', 'c', 'dt', 'frames', 'sigma', 'mu', 'tau'])
        info_values = np.array([self.N, self.q, self.m, self.r_max, self.t_max, self.k, self.c, self.dt, self.frames, self.sigma, self.mu, self.tau])

        np.savez(
            f'{directory}/solved/{filename}',
            info_keys=info_keys,
            info_values=info_values,
            data=self.data
        )
