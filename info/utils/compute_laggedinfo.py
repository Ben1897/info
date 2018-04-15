"""
This script is used for computing the coupling strengths (MI or CMI) among components
with a range of lags.
"""

import numpy as np
import pandas as pd
from .sst import independence, conditionalIndependence
from .others import reorganize_data, dropna
from ..core.info import computeMIKNN, computeCMIKNN


def compute_couplestrength_mi(data, k=5, tau=5):
    """Compute the coupling strengths based on mutual information without significance test.

       Inputs:
       data       -- the data points [numpy array with shape (npoints, ndim)]
    """
    # Get the dimension of the data
    npts, ndim = data.shape

    # Initialize the output
    miset = np.zeros([ndim, ndim, tau])  # mi[X,Y,tau]: X[t] and Y[t+tau]

    for i in range(ndim):
        for j in range(ndim):
            for t in range(tau):
                node1, node2 = [i,0], [j,t]
                # reorganize the data
                dataij = reorganize_data(data, [node1, node2])

                # Drop the rows with nan values
                dataijn = dropna(dataij)

                # Calculate the mutual information of them
                mi = computeMIKNN(data=dataijn, k=k)

                miset[i,j,t] = mi

    return miset


def compute_couplestrength_mi_sst(data, k=5, tau=5, ntest=100, alpha=0.05):
    """Compute the coupling strengths based on mutual information with significance test.

       Inputs:
       data       -- the data points [numpy array with shape (npoints, ndim)]
    """
    # Get the dimension of the data
    npts, ndim = data.shape

    # Initialize the output
    miset  = np.zeros([ndim, ndim, tau])  # mi[X,Y,tau]: X[t] and Y[t+tau]
    sigset = np.zeros([ndim, ndim, tau])
    upset  = np.zeros([ndim, ndim, tau])
    lowset = np.zeros([ndim, ndim, tau])

    for i in range(ndim):
        for j in range(ndim):
            for t in range(tau):
                node1, node2 = [i,0], [j,t]
                sig, mi, up, low = independence(node1, node2, data, shuffle_ind=[0], ntest=ntest, alpha=alpha,  approach='knn',
                                                k=k, returnTrue=True)

                miset[i,j,t]  = mi
                sigset[i,j,t] = sig
                upset[i,j,t]  = up
                lowset[i,j,t] = low

    return sigset, miset, upset, lowset


def compute_couplestrength_cmi(data, k=5, tau=5):
    """Compute the coupling strengths based on conditional mutual information without significance test.

       Inputs:
       data       -- the data points [numpy array with shape (npoints, ndim)]
    """
    pass


def compute_couplestrength_cmi_sst(data, k=5, tau=5, ntest=100, alpha=0.05):
    """Compute the coupling strengths based on conditional mutual information with significance test.

       Inputs:
       data       -- the data points [numpy array with shape (npoints, ndim)]
    """
    pass
