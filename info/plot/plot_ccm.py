"""
A set of functions for plotting results from CCM.

@Author: Peishi Jiang <Ben1897>
@Email:  shixijps@gmail.com

plot_ccm_with_trajectory()
plot_ccm_est_obs_xy()
plot_ccm_3d_xy()
plot_extended_ccm_xy()
plot_extended_ccm_3d()

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import matplotlib.gridspec as gridspec


def plot_ccm_with_trajectory(x_set, y_set, lag, L_set, xmpy, ympx, rhoset):
    """
    Return a complex plot based on CCM results.

    (1) The trajectory of x and y with lag.
    (2) The CCM skills in terms of different time series lengths.

    """
    plt.rcParams["figure.figsize"] = (20, 20)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.4, hspace=0.4)

    ax = plt.subplot(gs[0, 0])
    ax.scatter(x_set[:-lag], x_set[lag:])
    ax.set_xlabel('X(t)')
    ax.set_ylabel('X(t-1)')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    ax = plt.subplot(gs[0, 1])
    ax.scatter(y_set[:-lag], y_set[lag:])
    ax.set_xlabel('Y(t)')
    ax.set_ylabel('Y(t-1)')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    ax = plt.subplot(gs[1, :])
    ax.plot(L_set, xmpy, 'b', label='x xmp y')
    ax.plot(L_set, ympx, 'r', label='y xmp x')
    ax.plot(L_set, rhoset, 'k', label='cross-correlation')
    ax.set_xlabel('Length of the time series')
    ax.set_ylabel('Correlation coefficient')
    ax.set_ylim([-1, 1])
    ax.legend(loc='lower right')


def plot_ccm_est_obs_xy(x_obs, x_est, y_obs, y_est, rhox, rhoy):
    """Plot the observation versus the estimation from ccm for x and y."""
    plt.rcParams["figure.figsize"] = (20, 8)

    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.4, hspace=0.4)

    ax = plt.subplot(gs[0, 0])
    ax.plot(y_obs, y_est, '.')
    ax.set_xlabel('Y(t) observed')
    ax.set_ylabel('Y(t) estimated')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('rho: %.2f' % rhoy)

    ax = plt.subplot(gs[0, 1])
    ax.plot(x_obs, x_est, '.')
    ax.set_xlabel('X(t) observed')
    ax.set_ylabel('X(t) estimated')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('rho: %.2f' % rhox)


def plot_ccm_3d_xy(xmpy, ympx, xv, yv, xlabel, ylabel):
    """Plot the CCM skills for both xmpy and ympx in terms of two parameters xlabel and ylabel."""
    plt.rcParams["figure.figsize"] = (24, 8)

    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.4, hspace=0.4)

    ax = plt.subplot(gs[0, 0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, xmpy, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('xmpy')

    ax = plt.subplot(gs[0, 1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, ympx, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('ympx')


def plot_extended_ccm_xy(lagset, rhoxmpy, rhoympx, rhoxy):
    """Plot the correlation coefficients and the CCM skills in terms of different lags for 2 nodes."""
    # Get the lags with the maximum skill
    maxympx, maxxmpy = lagset[np.argmax(rhoympx)], lagset[np.argmax(rhoxmpy)]

    # Plot
    plt.rcParams["figure.figsize"] = (10, 6)

    plt.plot(lagset, rhoxmpy, 'b', label='x xmp y')
    plt.plot(lagset, rhoympx, 'r', label='y xmp x')
    plt.plot(lagset, rhoxy, 'k', label='rho(x, y)')
    plt.xlabel('Lag')
    plt.ylabel('correlation coefficient')
    plt.ylim([-0.5, 1.1])
    plt.title('Max. lag (x xmp y): %d; Max. lag(y xmp x): %d.' % (maxxmpy, maxympx))
    plt.legend(loc='lower right')


def plot_extended_ccm_3d(lagset, rhoxmpy, rhoympx,
                         rhoympz, rhozmpy, rhoxmpz, rhozmpx,
                         rhoxy, rhoyz, rhoxz):
    """Plot the correlation coefficients and the CCM skills in terms of different lags for 3 nodes."""
    # Get the lags with the maximum skill
    maxympx, maxxmpy = lagset[np.argmax(rhoympx)], lagset[np.argmax(rhoxmpy)]
    maxympz, maxzmpy = lagset[np.argmax(rhoympz)], lagset[np.argmax(rhozmpy)]
    maxxmpz, maxzmpx = lagset[np.argmax(rhoxmpz)], lagset[np.argmax(rhozmpx)]

    # Plot
    plt.rcParams["figure.figsize"] = (10, 20)
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.4, hspace=0.4)

    ax = plt.subplot(gs[0, 0])
    ax.plot(lagset, rhoxmpy, 'b', label='x xmp y')
    ax.plot(lagset, rhoympx, 'r', label='y xmp x')
    ax.plot(lagset, rhoxy, 'k', label='rho(x, y)')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-.5, 1.1])
    ax.set_title('Max. lag (x xmp y): %d; Max. lag(y xmp x): %d.' % (maxxmpy, maxympx))
    ax.legend(loc='lower right')

    ax = plt.subplot(gs[1, 0])
    ax.plot(lagset, rhoympz, 'b', label='y xmp z')
    ax.plot(lagset, rhozmpy, 'r', label='z xmp y')
    ax.plot(lagset, rhoyz, 'k', label='rho(y, z)')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-.5, 1.1])
    ax.set_title('Max. lag (y xmp z): %d; Max. lag(z xmp y): %d.' % (maxympz, maxzmpy))
    ax.legend(loc='lower right')

    ax = plt.subplot(gs[2, 0])
    ax.plot(lagset, rhoxmpz, 'b', label='x xmp z')
    ax.plot(lagset, rhozmpx, 'r', label='z xmp x')
    ax.plot(lagset, rhoxz, 'k', label='rho(x, z)')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-.5, 1.1])
    ax.set_title('Max. lag (x xmp z): %d; Max. lag(z xmp x): %d.' % (maxxmpz, maxzmpx))
    ax.legend(loc='lower right')


def plot_extended_ccm_3d_withmi(lagset, rhoxmpy, rhoympx,
                                rhoympz, rhozmpy, rhoxmpz, rhozmpx,
                                rhoxy, rhoyz, rhoxz,
                                mirxy, mirxz, miryz):
    """Plot the correlation coefficients and the CCM skills in terms of different lags for 3 nodes."""
    # Get the lags with the maximum skill
    maxympx, maxxmpy = lagset[np.argmax(rhoympx)], lagset[np.argmax(rhoxmpy)]
    maxympz, maxzmpy = lagset[np.argmax(rhoympz)], lagset[np.argmax(rhozmpy)]
    maxxmpz, maxzmpx = lagset[np.argmax(rhoxmpz)], lagset[np.argmax(rhozmpx)]

    # Plot
    plt.rcParams["figure.figsize"] = (10, 20)
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.4, hspace=0.4)

    xyt = mirxy==True
    xyf = mirxy==False
    ax = plt.subplot(gs[0, 0])
    ax.plot(lagset, rhoxmpy, 'b', label='x xmp y')
    ax.plot(lagset, rhoympx, 'r', label='y xmp x')
    ax.plot(lagset, rhoxy, 'k', label='rho(x, y)')
    ax.scatter(lagset[xyt], np.ones(xyt.sum()), color='red', label='MI SST True')
    ax.scatter(lagset[xyf], np.ones(xyf.sum()), color='blue', label='MI SST False')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-.5, 1.1])
    ax.set_title('Max. lag (x xmp y): %d; Max. lag(y xmp x): %d.' % (maxxmpy, maxympx))
    ax.legend(loc='lower right')

    yzt = miryz==True
    yzf = miryz==False
    ax = plt.subplot(gs[1, 0])
    ax.plot(lagset, rhoympz, 'b', label='y xmp z')
    ax.plot(lagset, rhozmpy, 'r', label='z xmp y')
    ax.plot(lagset, rhoyz, 'k', label='rho(y, z)')
    ax.scatter(lagset[yzt], np.ones(yzt.sum()), color='red', label='MI SST True')
    ax.scatter(lagset[yzf], np.ones(yzf.sum()), color='blue', label='MI SST False')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-.5, 1.1])
    ax.set_title('Max. lag (y xmp z): %d; Max. lag(z xmp y): %d.' % (maxympz, maxzmpy))
    ax.legend(loc='lower right')

    xzt = mirxz==True
    xzf = mirxz==False
    ax = plt.subplot(gs[2, 0])
    ax.plot(lagset, rhoxmpz, 'b', label='x xmp z')
    ax.plot(lagset, rhozmpx, 'r', label='z xmp x')
    ax.plot(lagset, rhoxz, 'k', label='rho(x, z)')
    ax.scatter(lagset[xzt], np.ones(xzt.sum()), color='red', label='MI SST True')
    ax.scatter(lagset[xzf], np.ones(xzf.sum()), color='blue', label='MI SST False')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-.5, 1.1])
    ax.set_title('Max. lag (x xmp z): %d; Max. lag(z xmp x): %d.' % (maxxmpz, maxzmpx))
    ax.legend(loc='lower right')
