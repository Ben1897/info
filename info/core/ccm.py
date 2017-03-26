# A python code for implementing CCM
#
# Ref:
# [1] Sugihara, George, et al. "Detecting causality in complex ecosystems." science 338.6106 (2012): 496-500.
# [2] Ye, Hao, et al. "Distinguishing time-delayed causal interactions using convergent cross mapping." Scientific reports 5 (2015).

import numpy as np
from sklearn.neighbors import NearestNeighbors

def ccm(x, y, x_future, y_future, nemb, tau=1, nn=None, scoremethod='corr', filtered=False):
    '''
    Implementation of the convergent cross mapping method.
    Inputs:
    x           -- the target variable [ndarray(npt,)]
    y           -- the source variable [ndarray(npt,)]
    x_future    -- the future values of the target variable [ndarray(npt2,)]
    y_future    -- the fugure values of the source variable [ndarray(npt2,)]
    nemb        -- the embedded dimension [int]
    tau         -- the time lag [int]
    nn          -- the number of the nearest neighbors [int]
    scoremethod -- the scoring method [string]
    filtered    -- a boolean value decide using find_knn or find_knn2 [boolean]
    Outputs:
    y_est -- the estimated source variable [ndarray(nptn,)]
    rho  -- the correlation coefficient between the y_est and y [float]
    '''
    if nn is None:
        nn = nemb + 1

    # Check whether x and y have the same length
    if x.size != y.size:
        raise Exception('The lengths of x and y are not the same!')
    # Check whether x_future and y_future have the same length
    if x_future.size != y_future.size:
        raise Exception('The lengths of x_future and y_future are not the same!')

    # Get the length of the time series and
    npt2 = x_future.size
    npt  = x.size

    # Get the index of the first x in the shadow manifold
    ind0  = tau*(nemb-1)

    # Create the shadow manifold
    nptn, x_man         = create_shadow_manifold(x, tau, nemb)
    nptn2, x_future_man = create_shadow_manifold(x_future, tau, nemb)

    # Find the indices and the k-nearest neighbours and the corresponding distances
    if filtered:
        dist, indices = find_knn2(x_man, x_future_man, nn)
    else:
        dist, indices = find_knn(x_man, x_future_man, nn)

    # Calculate the exponentially weighted distances
    distw = exp_d(dist)

    # Normalize the distw
    w = create_weights(distw)

    # Estimate y (i.e., get y_est)
    y = y[ind0:]
    y_est = predict(w, indices, y)

    # Compute the correlation coefficient between y_est and y
    y_true = y_future[ind0:]
    rho = score(y_est, y_true, method=scoremethod) 

    return y_est, rho


def extended_ccm(x, y, x_future, y_future, nemb, lag, tau=1, nn=None, filtered=False):
    '''
    Implementation of the convergent cross mapping method.
    Inputs:
    x     -- the target variable [ndarray(npt,)]
    y     -- the source variable [ndarray(npt,)]
    nemb  -- the embedded dimension [int]
    lag   -- the maximum lag between x and y [int]
    tau   -- the time lag [int]
    nn    -- the number of the nearest neighbors [int]
    filtered    -- a boolean value decide using find_knn or find_knn2 [boolean]
    Outputs:
    lagset  -- the set of the lags between x and y
    rhoxmpy  -- the correlation coefficients from x mp y in terms of
                different lag in lagset [ndarray(lag*2+1)]
    rhoympx  -- the correlation coefficients from y mp x in terms of
                different lag in lagset [ndarray(lag*2+1)]
    '''
    # TODO: Check whether the maximum lag exceeds the lengths of x and y
    lagset = np.arange(-lag, lag+1, 1, dtype='int')
    lagsetsize = lagset.size

    rhoxmpy = np.zeros(lagsetsize)
    rhoympx = np.zeros(lagsetsize)

    # Conduct CCM for different lags in lagset
    for i in range(lagsetsize):
        current_lag  = lagset[i]

        # Get the x and y give the lag
        if current_lag > 0:
            x_adj, y_adj = x[:-current_lag], y[current_lag:]
            x_future_adj, y_future_adj = x_future[:-current_lag], y_future[current_lag:]
        elif current_lag < 0:
            x_adj, y_adj = x[-current_lag:], y[:current_lag]
            x_future_adj, y_future_adj = x_future[-current_lag:], y_future[:current_lag]
        elif current_lag == 0:
            x_adj, y_adj = x, y
            x_future_adj, y_future_adj = x_future, y_future

        # Conduct CCM
        _, rho1 = ccm(x_adj, y_adj, x_future_adj, y_future_adj, nemb, tau, nn, filtered=filtered)  # x[t] xmp y[t+lag] 
        _, rho2 = ccm(y_adj, x_adj, y_future_adj, x_future_adj, nemb, tau, nn, filtered=filtered)  # y[t] xmp x[t-lag]

        rhoxmpy[i] = rho1
        rhoympx[-i-1] = rho2

    return lagset, rhoxmpy, rhoympx


def create_shadow_manifold(x, tau, nemb):
    '''
    Create the shadow manifold given a time series, the lag time and the embedded
    dimension.
    Inputs:
    x    -- the target variable [ndarray(npt,)]
    nemb -- the embedded dimension [int]
    tau  -- the time lag [int]
    Outputs:
    nptn  --  the length of the shadow manifold [int]
    x_man -- the shadow manifold of x [ndarray(nptn, nemb)]
    '''
    # Get the index of the first x in the shadow manifold
    ind0 = tau*(nemb-1)

    # Get the length of the shadow manifold
    nptn = x.size - ind0
    if nptn < 0:
        raise Exception('The length of x is smaller than the required embedded range!')

    # Create the shadow manifold
    x_man = np.zeros([nptn, nemb])
    for i in range(nptn):
        x_man[i] = x[i:i+ind0+1:tau]

    return nptn, x_man

def find_knn(x_man, x_future_man, nn):
    '''
    Find the k-nearest-neighbors.
    Inputs:
    x_man        -- the shadow manifold of x [ndarray(nptn, nemb)]
    x_future_man -- the shadow manifold of the future x [ndarray(nptn2, nemb)]
    nn           -- the number of the nearest neighbors
    Outputs:
    dist         -- the distances between the neighbors and each x_future_man
                    [ndarray(nptn2, nn)]
    indices      -- the indices of the neighbors from x_man in each x_future_man
                    [ndarray(nptn2, nn)]
    '''
    nptn2, nemb = x_future_man.shape

    if x_man.shape == x_future_man.shape and (x_man==x_future_man).all():
        nbrs          = NearestNeighbors(n_neighbors=nn+1, algorithm='kd_tree').fit(x_man)
        dist, indices = nbrs.kneighbors(x_future_man)
        dist, indices = dist[:, 1:], indices[:, 1:]  # exclude the index of the point itself
    else:
        nbrs          = NearestNeighbors(n_neighbors=nn, algorithm='kd_tree').fit(x_man)
        dist, indices = nbrs.kneighbors(x_future_man)

    dist = dist + 1e-6
    return dist, indices


def find_knn2(x_man, x_future_man, nn):
    '''
    Find the k-nearest-neighbors.
    Inputs:
    x_man        -- the shadow manifold of x [ndarray(nptn, nemb)]
    x_future_man -- the shadow manifold of the future x [ndarray(nptn2, nemb)]
    nn           -- the number of the nearest neighbors
    Outputs:
    dist         -- the distances between the neighbors and each x_future_man
                    [ndarray(nptn2, nn)]
    indices      -- the indices of the neighbors from x_man in each x_future_man
                    [ndarray(nptn2, nn)]
    '''
    nptn2, nemb = x_future_man.shape

    # Create an empty matrices for distances and indices
    dist    = np.zeros([nptn2, nn], dtype='float64')
    indices = np.zeros([nptn2, nn], dtype='int')

    # Calculate the distance and indices for each vector in x_future_man
    # Note: for each vector in x_future_man, exclude all the vectors in
    #       x_man which share the coordinates with it.
    for i in range(nptn2):
        # Get the current vector in x_future_man
        v_future = x_future_man[i]

        # Construct x_man_temp by excluding all the vectors sharing with v_future
        judge = np.array(map(lambda x: ~np.in1d(v_future, x).any(), x_man))
        # print judge, judge.shape
        x_man_temp = x_man[judge]

        # Conduct the knn
        nbrs = NearestNeighbors(n_neighbors=nn, algorithm='kd_tree').fit(x_man_temp)
        distv, indicesv = nbrs.kneighbors([v_future])

        dist[i]    = distv[0]
        indices[i] = indicesv[0]

    dist = dist + 1e-6
    return dist, indices


def exp_d(dist):
    '''
    Convert the distance into the exponential form.
    Inputs:
    dist -- the original distance [ndarray(nrow, ncol)]
    Outputs:
    distw -- the exponential form of the distance [ndarry(nrow, ncol)]
    '''
    nrow, ncol = dist.shape

    distmin = np.min(dist, axis=1)[:, np.newaxis]  # the smallest distance in each vector
    # distmin = np.tile(distmin, (ncol, 1)).T
    distw = np.exp(-dist / distmin)

    return distw


def create_weights(dist):
    '''
    Create the weights based on the distance and indices.
    Inputs:
    dist -- the original distance [ndarray(nrow, ncol)]
    Outputs:
    w    -- the weights [ndarray(nrow, ncol)]
    '''
    nrow, ncol = dist.shape

    w = dist / np.sum(dist, axis=1)[:, np.newaxis]
    # w[np.where(np.isnan(w))] = 1./ncol   # convert all the nan value to 1/ncol

    return w


def predict(w, indices, y):
    '''
    Make a prediction based on the weights
    Inputs:
    w       -- the weights [ndarray(nrow, ncol)]
    y       -- the source variable [ndarray(nrow2,)]
    indices -- the indices of the neighbors from x_man in each x_future_man
               [ndarray(nrow, ncol)]
    Outputs:
    y_est -- the estimated source variable [ndarray(nrow,)]
    '''
    nrow, ncol = w.shape

    y_est  = np.zeros(nrow)
    for i in range(nrow):
        y_est[i] = np.sum(w[i] * y[indices[i]])

    return y_est


def score(y_est, y_true, method='corr'):
    '''
    Calculate the estimation score by comparing the true values
    Inputs:
    y_est  -- the estimated source variable [ndarray(nrow,)]
    y_true -- the true source variable [ndarray(nrow,)]
    Outputs:
    rho    -- the correlation coefficient
    '''
    if method == 'corr':
        rho = np.corrcoef(y_est, y_true)[0, 1]

    return rho
