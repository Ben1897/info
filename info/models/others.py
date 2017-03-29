"""
Several numerical models.

@Author: Peishi Jiang <Peishi>
@Email:  shixijps@gmail.com

"""

import numpy as np


def fishery_model(N):
    """
    A standard fishery model system with two non-interacting populations that share common environmental forcing.

    Ref: Eqn. (S4) in Sugihara, George, et al. "Detecting causality in complex ecosystems." science 338.6106 (2012): 496-500.

    """
    rxf = lambda x, t: x*(3.1*(1-x))*np.exp(-.3*t)
    ryf = lambda y, t: y*(2.9*(1-y))*np.exp(-.36*t)
    xf = lambda x, rx: 0.4*x+max(rx, 0)
    yf = lambda y, ry: 0.35*y+max(ry, 0)

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
