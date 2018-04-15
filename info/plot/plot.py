"""
A set of plotting functions for plotting results of coupling strength or cumulative information transfer
"""

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import matplotlib.gridspec as gridspec


def plot_lagged_cs(csset, varnames, sigset=None, transformed=False, axes=None, label='MI',figsize=(18,12)):
    """ Plotting the lagged coupling strengths
    Inputs:
    csset    - a set of lagged coupling strengths [ndarray with shape (ndim, ndim, tau)]
    varnames - a list of variable names [list]
    """
    ndim, ndim, taumax = csset.shape

    if axes is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=figsize)

    if transformed:
        csset = np.sqrt(1-np.exp(-2.*csset))

    # Plot the coupling strength
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i,j]
            if sigset is None:
                ax.plot(range(taumax), csset[i,j,:], 'k.', label=label)
            else:
                cs, sig = csset[i,j,:], sigset[i,j,:]
                pos, neg = np.where(sig==True)[0], np.where(sig==False)[0]
                if pos.size:
                    ax.plot(pos, cs[pos], 'b.', label=label + ' (insig)')
                if neg.size:
                    ax.plot(neg, cs[neg], 'r.', label=label + ' (sig)')
            if transformed:
                ax.set_ylim([-.2, 1])

    # Set labels
    for i in range(ndim):
        ax = axes[i,0]
        ax.set_ylabel(varnames[i])

    for j in range(ndim):
        ax1, ax2 = axes[0,j], axes[-1,j]
        ax1.set_title("-> " + varnames[j])
        ax2.set_xlabel(varnames[j])


def plot_lagged_sig(sigset, varnames, axes=None, figsize=(18,12)):

    ndim, ndim, taumax = sigset.shape

    if axes is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=figsize)

    # Plot the coupling strength
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i,j]
            ax.plot(range(taumax), sigset[i,j,:], 'k.')

    # Set labels
    for i in range(ndim):
        ax = axes[i,0]
        ax.set_ylabel(varnames[i])

    for j in range(ndim):
        ax1, ax2 = axes[0,j], axes[-1,j]
        ax1.set_title("-> " + varnames[j])
        ax2.set_xlabel(varnames[j])
