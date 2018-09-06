"""
Several numerical models.

@Author: Peishi Jiang <Peishi>
@Email:  shixijps@gmail.com

two_species_logistic()
two_species_logistic_delayed()
fishery_model()
Lorenze_model()
common_driver_linear()
common_driver_nonlinear()

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


def Lorenz_model(N, dt, e=0., seed=1, init=None, s=10, r=28, b=2.667):
    """The Lorenz model."""
    if seed is not None:
        np.random.seed(1)
    else:
        np.random.seed(seed)

    trash = 2000
    data = np.zeros([trash+N+1, 3])

    def lorenzele(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    # Setting initial values
    if init is not None:
        data[0,:] = init
    else:
        data[0,:] = np.random.random(3)


    # Stepping through "time".
    for i in range(trash+N):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenzele(data[i,0], data[i,1], data[i,2], s=s, r=r, b=b)
        data[i+1,0] += data[i,0] + (x_dot * dt) + e*np.random.rand()*dt
        data[i+1,1] += data[i,1] + (y_dot * dt) + e*np.random.rand()*dt
        data[i+1,2] += data[i,2] + (z_dot * dt) + e*np.random.rand()*dt

    # print data[:10,:]
    return data[trash:]

def henon_map(N, e=0., seed=1, init=None, a=1.4, b=0.3):
    """The Lorenz model."""
    if seed is not None:
        np.random.seed(1)
    else:
        np.random.seed(seed)

    trash = 2000
    data = np.zeros([trash+N+1, 2])

    # Setting initial values
    if init is not None:
        data[0,:] = init
    else:
        data[0,:] = np.random.uniform(low=.25, high=.4, size=2)

    print data[0,:]

    # Stepping through "time".
    for i in range(trash+N):
        data[i+1,0] = 1 - a*data[i,0]**2 + data[i,1]  + e*np.random.rand()
        data[i+1,1] = b*data[i,0] + e*np.random.rand()

    # print data[:10,:]
    return data[trash:]

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


def common_driver_linear(N, cwx, cwy, cxz, cyz, sw=1, sy=1, sx=1, sz=1):
    '''
    A linear model representing two causes induced by a common driver.
    w(t) = ew
    y(t) = cwy*w(t-1) + ey
    x(t) = cwx*w(t-1) + ex
    z(t) = cxz*x(t-1) + cyz*y(t-1) + ez
    where ew, ey, ex, ez are white noises
    '''
    ew = lambda: np.random.normal(0, sw, 1)
    ey = lambda: np.random.normal(0, sy, 1)
    ex = lambda: np.random.normal(0, sx, 1)
    ez = lambda: np.random.normal(0, sz, 1)

    w, x, y, z = np.zeros(N+2), np.zeros(N+2), np.zeros(N+2), np.zeros(N+2)

    w[0] = ew(); w[1] = ew()
    x[1] = cwx*w[0] + ex(); y[1] = cwy*w[0] + ey()
    for i in range(2, N+2):
        w[i] = ew()
        y[i] = cwy*w[i-1] + ey()
        x[i] = cwx*w[i-1] + ex()
        z[i] = cxz*x[i-1] + cyz*y[i-1] + ez()

    return w[2:], x[2:], y[2:], z[2:]


def common_driver_nonlinear(N, cwx, cwy, cz, sw=1, sy=1, sx=1, sz=1):
    '''
    A nonlinear model representing two causes induced by a common driver.
    w(t) = ew
    y(t) = cwy*w(t-1) + ey
    x(t) = cwx*w(t-1) + ex
    z(t) = cz*x(t-1)*y(t-1) + ez
    where ew, ey, ex, ez are white noises
    '''
    ew = lambda: np.random.normal(0, sw, 1)
    ey = lambda: np.random.normal(0, sy, 1)
    ex = lambda: np.random.normal(0, sx, 1)
    ez = lambda: np.random.normal(0, sz, 1)

    w, x, y, z = np.zeros(N+2), np.zeros(N+2), np.zeros(N+2), np.zeros(N+2)

    w[0] = ew(); w[1] = ew()
    x[1] = cwx*w[0] + ex(); y[1] = cwy*w[0] + ey()
    for i in range(2, N+2):
        w[i] = ew()
        y[i] = cwy*w[i-1] + ey()
        x[i] = cwx*w[i-1] + ex()
        z[i] = cz*x[i-1]*y[i-1] + ez()

    return w[2:], x[2:], y[2:], z[2:]
