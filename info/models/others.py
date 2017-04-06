"""
Several numerical models.

@Author: Peishi Jiang <Peishi>
@Email:  shixijps@gmail.com

two_species_logistic()
two_species_logistic_delayed()
fishery_model()
Lorenze_model()

"""


import numpy as np


def two_species_logistic(x0, y0, N, rx, ry, bxy, byx):
    """A coupled two-species nonlinear logistic different system with chaotic dynamics."""
    x_set = np.zeros(N)
    y_set = np.zeros(N)
    x_set[0] = x0
    y_set[0] = y0

    for i in range(N-1):
        x_set[i+1] = x_set[i]*(rx-rx*x_set[i]-bxy*y_set[i])
        y_set[i+1] = y_set[i]*(ry-ry*y_set[i]-byx*x_set[i])

    return x_set, y_set


def two_species_logistic_delayed(x0, y0, N, rx, ry, bxy, byx, tau):
    """A coupled two-species nonlinear logistic difference system with chaotic dynamics with delays."""
    x_set = np.zeros(N)
    y_set = np.zeros(N)
    x_set[0:tau+1] = x0
    y_set[0:tau+1] = y0

    for i in range(N-1-tau):
        x_set[i+1+tau] = x_set[i+tau]*(rx-rx*x_set[i+tau]-bxy*y_set[i+tau])
        y_set[i+1+tau] = y_set[i+tau]*(ry-ry*y_set[i+tau]-byx*x_set[i])

    return x_set, y_set


def Lorenze_model(x0, y0, z0, N, dt, sigma, r, b):
    """The Lorenze model."""
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    for i in range(N-1):
        x[i+1] = x[i] + dt*(sigma*(y[i]-x[i]))
        y[i+1] = y[i] + dt*(-x[i]*z[i] + r*x[i] - y[i])
        z[i+1] = z[i] + dt*(x[i]*y[i] - b*z[i])

    return x, y, z


def fishery_model(N, rxx=3.1, rxt=-.3, ryy=2.9, ryt=-.36, cx=.4, cy=.35):
    """
    A standard fishery model system with two non-interacting populations that share common environmental forcing.

    Ref: Eqn. (S4) in Sugihara, George, et al. "Detecting causality in complex ecosystems." science 338.6106 (2012): 496-500.

    """
    rxf = lambda x, t: x*(rxx*(1-x))*np.exp(rxt*t)
    ryf = lambda y, t: y*(ryy*(1-y))*np.exp(ryt*t)
    xf = lambda x, rx: cx*x+max(rx, 0)
    yf = lambda y, ry: cy*y+max(ry, 0)

    trash = 100
    x_set = np.zeros(N+trash)
    y_set = np.zeros(N+trash)
    rx_set = np.zeros(N+trash)
    ry_set = np.zeros(N+trash)
    t_set = np.zeros(N+trash)

    rx_set[0] = np.random.rand()
    ry_set[0] = np.random.rand()
    x_set[:4] = np.random.rand(4)
    y_set[:4] = np.random.rand(4)

    for i in range(3):
        t = np.random.normal()
        xnow, ynow = x_set[i], y_set[i]
        t_set[i] = t
        rx_set[i+1] = rxf(xnow, t)
        ry_set[i+1] = ryf(ynow, t)

    for i in range(4, N+trash-1):
        xnow, ynow = x_set[i-1], y_set[i-1]
        rxnow, rynow = rx_set[i-4], ry_set[i-4]
        t = np.random.normal()
        t_set[i-1] = t
        x_set[i] = xf(xnow, rxnow)
        y_set[i] = yf(xnow, rynow)
        rx_set[i] = rxf(xnow, t)
        ry_set[i] = ryf(ynow, t)

    t_set[-1] = np.random.normal()

    return x_set[trash:], y_set[trash:], t_set[trash:]
