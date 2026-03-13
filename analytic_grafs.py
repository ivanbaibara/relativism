import numpy as np
from scipy.optimize import root
from scipy.special import erf
from numpy import exp, log, sqrt, pi, arccosh
import warnings
from constants import *


# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
Np = 1e3
m = 0.0
s = 0.2
Rmax = R0
Tmax = T0
Qtotal = Q0
Mtotal = M0
e0 = eps0

alpha = Qtotal / Mtotal


# ФУНКЦИИ
def rho_n(r):
    return exp(-(log(r) - m) ** 2 / (2 * s ** 2)) / (sqrt(2 * pi) * s * r)

def rho(r):
    return rho_n(tau * r) * tau * Qtotal / (4 * pi * r ** 2)

def Q_my(r):
    return (Qtotal / 2) * (1 + erf( (log(tau * r) - m) / (sqrt(2) * s) ) )

def gamma(r):
    return alpha * Q_my(r) / (4 * pi * e0)

def bb(r):
    return sqrt(gamma(r)) / c

def nn(r):
    return bb(r) / r ** 0.5

def lambda_(r):
    return c * nn(r) * sqrt(2 + nn(r) ** 2) / (r * (1 + nn(r) ** 2))

def db_my(r):
    return sqrt(alpha * pi / (c ** 2 * e0)) * (r ** 2 * rho(r)) / (sqrt(Q_my(r)))

def dn_my(r):
    return (2 * db_my(r) * r - bb(r)) / (2 * r ** 1.5)

def dlambda_my(r):
    return c * (2 * db_my(r) * r - bb(r) - nn(r) * sqrt(r) * (1 + nn(r) ** 2) * (2 + nn(r) ** 2)) / (r ** 2.5 * (1 + nn(r) ** 2) ** 2 * sqrt(2 + nn(r) ** 2))

def F(s, n):
    return sqrt((s - 1) * (s - n ** 2 / (2 + n ** 2))) + arccosh(s * (2 + n ** 2) - 1 - n ** 2) / ((2 + n ** 2) * (1 + n ** 2))

def dFx(s, n):
    return 1 / sqrt((s - 1) * (s - n ** 2 / (n ** 2 + 2))) * (1 + (1 + n ** 2) * (s + (s - 1) * (n ** 2 + 1))) / ((1 + n ** 2) * (2 + n ** 2))

def dFn(s, n):
    n2 = n ** 2
    term1 = -2 * n / ((2 + n2) ** 2 * (1 + n2))
    term2 = n2 * sqrt((s - 1) / (s - n2 / (n2 + 2)))
    term3 = (2 * n2 + 3) * arccosh(s * (2 + n2) - 1 - n2) / (1 + n2)
    return term1 * (term2 + term3)

def P(t, r0):
    solution = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        F1 = lambda x: F(x, nn(r0)) - t
        solution = root(F1, 1.01)

    return solution.x[0]

def FF(x, r, t):
    return F(r / x, nn(x)) - t * lambda_(x)

def dFF(x, r, t):
    return -dFx(r / x, nn(x)) * r / x ** 2  +  dFn(r / x, nn(x)) * dn_my(x) - dlambda_my(x) * t

def Rho(r, t):
    P_ = P(lambda_(r) * t, r)
    return rho(r) / (P_ + r * ( (dlambda_my(r) * t - dFn(P_, nn(r)) * dn_my(r)) / (dFx(P_, nn(r))) )) / P_ ** 2

def Rmy(r, t):
    return r * P(lambda_(r) * t, r)

