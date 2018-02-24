'''
Construct the time series graph for the network from observation data.
Because of the stationarity assumption, we return the parents of all the variables at a given time step.

Author: Peishi Jiang
Date: 2017-02-24

'''

import numpy as np

def findCausalRelationships(data, dtau, taumax):
    """
    Return the causal relationships among the variables or the parents of all the variables in a given time t in the time series graph.

    Inputs:
    data   -- the observation data [numpy array with shape (npoints, ndim)]
    dtau   -- the range of the time lags used for convergence check [int]
    taumax -- the maximum time lag for updating parents (also used for convergence check) [int]

    """

    # Initialize the graph, the parents set, and the time lag

    # Update the parents at time lag tau

    # Once the graph converges, assign the parents of each node at the last time step to the parents set

    # Return the parents set

    pass
