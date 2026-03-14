import numpy as np
from scipy.optimize import root
from scipy.special import erf
from numpy import exp, log, sqrt, pi, arccosh
import warnings


class ASolution:
    def __init__(self):
        self.c = None
        self.N = None
        self.mu = None
        self.tau = None
        self.sigma = None
        self.r_max = None
        self.t_max = None
        self.charge = None
        self.mass = None
        self.e0 = None
        self.alpha = None

    def set_constants(self, N, charge, mass, r_max, t_max, c, e0, sigma, mu, tau):
        self.N = N
        self.charge = charge
        self.mass = mass
        self.r_max = r_max
        self.t_max = t_max
        self.c = c
        self.e0 = e0
        self.sigma = sigma
        self.mu = mu
        self.tau = tau
        self.alpha = self.charge / self.mass

    def rho_n(self, r):
        return exp(-(log(r) - self.mu) ** 2 / (2 * self.sigma ** 2)) / (sqrt(2 * pi) * self.sigma * r)

    def rho(self, r):
        return self.rho_n(self.tau * r) * self.tau * self.charge / (4 * pi * r ** 2)

    def Q_my(self, r):
        return (self.charge / 2) * (1 + erf((log(self.tau * r) - self.mu) / (sqrt(2) * self.sigma)))

    def gamma(self, r):
        return self.alpha * self.Q_my(r) / (4 * pi * self.e0)

    def bb(self, r):
        return sqrt(self.gamma(r)) / self.c

    def nn(self, r):
        return self.bb(r) / r ** 0.5

    def lambda_(self, r):
        return self.c * self.nn(r) * sqrt(2 + self.nn(r) ** 2) / (r * (1 + self.nn(r) ** 2))

    def db_my(self, r):
        return sqrt(self.alpha * pi / (self.c ** 2 * self.e0)) * (r ** 2 * self.rho(r)) / (sqrt(self.Q_my(r)))

    def dn_my(self, r):
        return (2 * self.db_my(r) * r - self.bb(r)) / (2 * r ** 1.5)

    def dlambda_my(self, r):
        return self.c * (2 * self.db_my(r) * r - self.bb(r) - self.nn(r) * sqrt(r) * (1 + self.nn(r) ** 2) * (2 + self.nn(r) ** 2)) / (
                    r ** 2.5 * (1 + self.nn(r) ** 2) ** 2 * sqrt(2 + self.nn(r) ** 2))

    @staticmethod
    def F(s, n):
        return sqrt((s - 1) * (s - n ** 2 / (2 + n ** 2))) + arccosh(s * (2 + n ** 2) - 1 - n ** 2) / (
                    (2 + n ** 2) * (1 + n ** 2))

    @staticmethod
    def dFx(s, n):
        return 1 / sqrt((s - 1) * (s - n ** 2 / (n ** 2 + 2))) * (1 + (1 + n ** 2) * (s + (s - 1) * (n ** 2 + 1))) / (
                    (1 + n ** 2) * (2 + n ** 2))

    @staticmethod
    def dFn(s, n):
        n2 = n ** 2
        term1 = -2 * n / ((2 + n2) ** 2 * (1 + n2))
        term2 = n2 * sqrt((s - 1) / (s - n2 / (n2 + 2)))
        term3 = (2 * n2 + 3) * arccosh(s * (2 + n2) - 1 - n2) / (1 + n2)
        return term1 * (term2 + term3)

    def P(self, t, r0):
        solution = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            F1 = lambda x: self.F(x, self.nn(r0)) - t
            solution = root(F1, 1.01)

        return solution.x[0]

    def FF(self, x, r, t):
        return ASolution.F(r / x, self.nn(x)) - t * self.lambda_(x)

    def dFF(self, x, r, t):
        return -ASolution.dFx(r / x, self.nn(x)) * r / x ** 2 + ASolution.dFn(r / x, self.nn(x)) * self.dn_my(x) - self.dlambda_my(x) * t

    def Rho(self, r, t):
        P_ = self.P(self.lambda_(r) * t, r)
        return self.rho(r) / (P_ + r * ((self.dlambda_my(r) * t - ASolution.dFn(P_, self.nn(r)) * self.dn_my(r)) / (ASolution.dFx(P_, self.nn(r))))) / P_ ** 2

    def Rmy(self, r, t):
        return r * self.P(self.lambda_(r) * t, r)

    def get_function(self, points, t):
        xs = np.array([self.r_max * (0.1 + i / points) for i in range(points)])
        ys = None

        if t == 0:
            ys = np.array([self.rho(x) for x in xs])
        else:
            xn = np.array([self.Rmy(x, t) for x in xs])
            ys = np.array([self.Rho(x, t) for x in xs])

            xs = xn

        return xs, ys


