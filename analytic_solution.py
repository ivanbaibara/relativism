from scipy.optimize import root, brentq
from scipy.special import erf
from numpy import exp, log, sqrt, pi, arccosh, abs, arccos

class AnalyticSolutionGV:
    def __init__(self, sigma, m, tau, M0, R0, G, c):
        self.sigma = sigma
        self.m = m
        self.tau = tau
        self.M0 = M0
        self.R0 = R0
        self.G = G
        self.c = c

    def __rho_n(self, r):
        return exp(-(log(r) - self.m) ** 2 / (2 * self.sigma ** 2)) / (sqrt(2 * pi) * self.sigma * r)

    def rho(self, r):
        return self.tau * self.__rho_n(self.tau * r) * self.M0 / (4 * pi * r ** 2)

    def __Mmy(self, r):
        return self.M0 / 2 * (1 + erf((log(self.tau * r) - self.m) / (sqrt(2) * self.sigma)))

    def __gamma(self, r):
        return self.__Mmy(r) * self.G

    def __bb(self, r):
        return sqrt(self.__gamma(r)) / self.c

    def __nn(self, r):
        return self.__bb(r) / sqrt(r)

    def __lambda_(self, r):
        return (self.c * self.__nn(r) * sqrt(2 - self.__nn(r) ** 2)) / (r * (1 - self.__nn(r) ** 2))

    def __db_my(self, r):
        return (sqrt(self.G) * 2 * pi * r ** 2 * self.rho(r)) / (self.c * sqrt(self.__Mmy(r)))

    def __dn_my(self, r):
        return (2 * self.__db_my(r) * r - self.__bb(r)) / (2 * r ** 1.5)

    def __dlambda_my(self, r):
        return self.c * (2 * self.__db_my(r) * r - self.__bb(r) - self.__nn(r) * sqrt(r) * (1 - self.__nn(r) ** 2) * (2 - self.__nn(r) ** 2)) / (
                    r ** 2.5 * (1 - self.__nn(r) ** 2) ** 2 * sqrt(2 - self.__nn(r) ** 2))

    @staticmethod
    def __F(s, n):
        return sqrt((1 - s) * (s + n ** 2 / (2 - n ** 2))) + arccos(s * (2 - n ** 2) - 1 + n ** 2) / (
                    (2 - n ** 2) * (1 - n ** 2))

    @staticmethod
    def __dFx(s, n):
        return -1 / sqrt((1 - s) * (s + n ** 2 / (2 - n ** 2))) * (1 + (1 - n ** 2) * (s - (1 - s) * (1 - n ** 2))) / (
                    (1 - n ** 2) * (2 - n ** 2))

    @staticmethod
    def __dFn(s, n):
        n2 = n ** 2
        term1 = 2 * n / ((2 - n2) ** 2 * (1 - n2))
        term2 = -n2 * sqrt((1 - s) / (s + n2 / (2 - n2)))
        term3 = (3 - 2 * n2) * arccos(s * (2 - n2) - 1 + n2) / (1 - n2)
        return term1 * (term2 + term3)

    def __P2_solver(self, t, r0):
        Fs = lambda x: AnalyticSolutionGV.__F(x, self.__nn(r0)) - t
        res = root(Fs, 0.999)
        return res.x[0]

    def Rmy(self, t, r0):
        return r0 * self.__P2_solver(self.__lambda_(r0) * t, r0)

    def Rho(self, r, t):
        P_ = self.__P2_solver(self.__lambda_(r) * t, r)
        return self.rho(r) / (P_ + r * ((self.__dlambda_my(r) * t - AnalyticSolutionGV.__dFn(P_, self.__nn(r)) * self.__dn_my(r)) / AnalyticSolutionGV.__dFx(P_, self.__nn(r)))) / (P_ ** 2)
