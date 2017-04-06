"""
A function for conducting the statistical significance test on the mutual information of two variables.

@Author: Peishi Jiang <Ben1897>
@Date:   2017-02-17T10:47:00-06:00
@Email:  shixijps@gmail.com

conductSST_CCM_withlag()
conductSST_CCM()
conductSST_withlag()
conductSST()

"""

import numpy as np

from .pdf_computer import pdfComputer
from ..core.info import info
from ..core.ccm import ccm


sstmethod_set = ['traditional', 'segments', 'seasonal']


def shuffle_x(x, sstmethod, n):
    """Shuffle x based on the shuffling method.

    Input:
    x         -- the data series of the variable to be shuffled [ndarray with shape (ndim, nsamples)]
    sstmethod -- the method used for generating the surrogate data [str]
    n         -- the number of the shuffled data sets [int]
    Output:
    x_shuffled_all -- all the shuffled data sets [ndarray with shape (ndim, n, nsamples)]

    """
    ndim, nsamples = x.shape

    x_shuffled_all = np.zeros([ndim, n, nsamples])

    # Create the original indices
    ind = np.arange(0, nsamples, dtype=int)

    # Get the indices of all the permutations
    if sstmethod == 'traditional':
        ind_p = map(lambda i: np.random.permutation(ind), range(n))
    elif sstmethod == 'segments':
        ind_set = np.random.randint(low=0, high=nsamples, size=n)
        ind_p = map(lambda i: np.roll(ind, -i), ind_set)
    elif sstmethod == 'seasonal':
        raise Exception('Not ready for the seasonal surrogates!')
    else:
        raise Exception('Unknown method %s' % sstmethod)

    for i in range(ndim):
        x_c = x[i]
        x_shuffled_all[i] = np.array(map(lambda j: x_c[ind_p[j]], range(n)))

    return x_shuffled_all


def conductSST_CCM_librarylengths(L_set, x, y, x_future=None, y_future=None, nemb=2, tau=1, nn=None, scoremethod='corr',
                                  filtered=False, sstmethod='traditional', ntest=100):
    """Conduct the statistical significant test on CCM based on different library lengths.

    (1) Generate ntest of surrogates for only x.
    (2) Calculate CCM skills for each library length
        The calculated rho is significant if rho > 95% of the surrogates and rho < 5% of the surrogates
    Input:
    L_set       -- the set of the considered library lengths [ndarray with shape (nl)]
    x           -- the data series of the target variable [ndarray with shape (nsamples)]
    y           -- the data series of the source variable [ndarray with shape (nsamples)]
    x_future    -- the future data series of x [ndarray with shape (nsamples2)]
    y_future    -- the future data series of y [ndarray with shape (nsamples2)]
    nemb        -- the embedded dimension [int]
    tau         -- the time lag [int]
    nn          -- the number of the nearest neighbors [int]
    scoremethod -- the scoring method [string]
    filtered    -- a boolean value decide using find_knn or find_knn2 [boolean]
    sstmethod   -- the method used for generating the surrogate data [str]
    ntest       -- the number of the shuffle time [int]
    Output:
    result_set -- whether the CCM skill is significant [boolean]
    rho_set    -- the correlation coefficient between the observed and the estimated values [float]
    lower_set  -- the lower threshold of the rho based on the surrogated data sets (5%) [float]
    upper_set  -- the upper threshold of the rho based on the surrogated data sets (95%) [float]

    """
    # Get the length of the set of the library lengths
    nl = len(L_set)

    # Check whether xdata and ydata have the same length and are 1D
    if x.ndim != 1 or y.ndim != 1:
        raise Exception('The dimension of the input data should be 1D!')
    if x.shape != y.shape:
        raise Exception('The length of the two data sets are not the same!')

    # Create empty sets for result_set, rho_set, lower_set, upper_set
    result_set = np.zeros(nl)
    rho_set    = np.zeros(nl)
    lower_set  = np.zeros(nl)
    upper_set  = np.zeros(nl)

    # Calculate the CCM score for different library lengths
    # if there is no x_future and y_future, then they are the same as x and y
    if x_future == None and y_future == None:
        x_future, y_future = np.copy(x), np.copy(y)
    for j in range(nl):
        L = L_set[j]
        _, rho = ccm(x[:L], y[:L], x_future[:L], y_future[:L], nemb, tau, nn, scoremethod, filtered)
        rho_set[j] = rho

    # Shuffle all the four time series data for ntest times
    x_shuffled_all = shuffle_x(x[np.newaxis, :], sstmethod, ntest)[0]

    # Conduct SST for every library length
    for j in range(nl):
        L = L_set[j]
        # Calculate the CCM score of each pair of (x_shuffled and y)
        rho_shuffled_all = []
        for i in range(ntest):
            # Get shuffled data
            x_temp, x_future_temp = x_shuffled_all[i][:L], x_future[:L]
            y_temp, y_future_temp = y[:L], y_future[:L]
            # Calculate the corresponding CCM skill
            _, rho_shuffled = ccm(x_temp, y_temp, x_future_temp, y_future_temp,
                                  nemb, tau, nn, scoremethod, filtered)
            rho_shuffled_all.append(rho_shuffled)

        # Calculate 95% and 5% percentiles
        upper = np.percentile(rho_shuffled_all, 95)
        lower = np.percentile(rho_shuffled_all, 5)

        # Determine whether rho is significant
        rho = rho_set[j]
        if rho > upper or rho < lower:
            result_set[j] = True
        else:
            result_set[j] = False

        # Return results
        lower_set[j] = lower
        upper_set[j] = upper

    return result_set, rho_set, lower_set, upper_set

    # # Shuffle all the four time series data for ntest times
    # data_all = np.array([x, y, x_future, y_future])
    # data_shuffled_all = shuffle_x(data_all, sstmethod, ntest)
    # x_shuffled_all, x_future_shuffled_all = data_shuffled_all[0], data_shuffled_all[2]
    # y_shuffled_all, y_future_shuffled_all = data_shuffled_all[1], data_shuffled_all[3]

    # # Conduct SST for every library length
    # for j in range(nl):
    #     L = L_set[j]
    #     # Calculate the CCM score of each pair of (x_shuffled and y)
    #     rho_shuffled_all = []
    #     for i in range(ntest):
    #         # Get shuffled data
    #         x, x_future = x_shuffled_all[i][:L], x_future_shuffled_all[i][:L]
    #         y, y_future = y_shuffled_all[i][:L], y_future_shuffled_all[i][:L]
    #         # Calculate the corresponding CCM skill
    #         _, rho_shuffled = ccm(x, y, x_future, y_future, nemb, tau, nn, scoremethod, filtered)
    #         rho_shuffled_all.append(rho_shuffled)

    #     # Calculate 95% and 5% percentiles
    #     upper = np.percentile(rho_shuffled_all, 95)
    #     lower = np.percentile(rho_shuffled_all, 5)

    #     # Determine whether rho is significant
    #     rho = rho_set[j]
    #     if rho > upper or rho < lower:
    #         result_set[j] = True
    #     else:
    #         result_set[j] = False

    #     # Return results
    #     lower_set[j] = lower
    #     upper_set[j] = upper

    # return result_set, rho_set, lower_set, upper_set


def conductSST_CCM(x, y, x_future=None, y_future=None, nemb=2, tau=1, nn=None, scoremethod='corr',
                   filtered=False, sstmethod='traditional', ntest=100):
    """Conduct the statistical significance test on CCM.

    The shuffling procedure will be conducted on all the x, y, x_future and y_future.
    The calculated rho is significant if rho > 95% of the surrogates and rho < 5% of the surrogates
    Input:
    x           -- the data series of the target variable [ndarray with shape (nsamples)]
    y           -- the data series of the source variable [ndarray with shape (nsamples)]
    x_future    -- the future data series of x [ndarray with shape (nsamples2)]
    y_future    -- the future data series of y [ndarray with shape (nsamples2)]
    nemb        -- the embedded dimension [int]
    tau         -- the time lag [int]
    nn          -- the number of the nearest neighbors [int]
    scoremethod -- the scoring method [string]
    filtered    -- a boolean value decide using find_knn or find_knn2 [boolean]
    sstmethod   -- the method used for generating the surrogate data [str]
    ntest       -- the number of the shuffle time [int]
    Output:
    result -- whether the CCM skill is significant [boolean]
    rho    -- the correlation coefficient between the observed and the estimated values [float]
    lower  -- the lower threshold of the rho based on the surrogated data sets (5%) [float]
    upper  -- the upper threshold of the rho based on the surrogated data sets (95%) [float]
    """
    # Check whether xdata and ydata have the same length and are 1D
    if x.ndim != 1 or y.ndim != 1:
        raise Exception('The dimension of the input data should be 1D!')
    if x.shape != y.shape:
        raise Exception('The length of the two data sets are not the same!')

    # Calculate the CCM score
    # if there is no x_future and y_future, then they are the same as x and y
    if x_future == None and y_future == None:
        x_future, y_future = np.copy(x), np.copy(y)
    _, rho = ccm(x, y, x_future, y_future, nemb, tau, nn, scoremethod, filtered)

    # Shuffle all the four time series data for ntest times
    x_shuffled_all = shuffle_x(x[np.newaxis, :], sstmethod, ntest)[0]

    # Calculate the CCM score of each pair of (x_shuffled and y)
    rho_shuffled_all = []
    for i in range(ntest):
        # Get shuffled data
        x_temp, x_future_temp = x_shuffled_all[i], x_future
        y_temp, y_future_temp = y, y_future
        # Calculate the corresponding CCM skill
        _, rho_shuffled = ccm(x_temp, y_temp, x_future_temp, y_future_temp,
                              nemb, tau, nn, scoremethod, filtered)
        rho_shuffled_all.append(rho_shuffled)

    # Calculate 95% and 5% percentiles
    upper = np.percentile(rho_shuffled_all, 95)
    lower = np.percentile(rho_shuffled_all, 5)

    # Return results:
    if rho > upper or rho < lower:
        return True, rho, upper, lower
    else:
        return False, rho, upper, lower

    # # Shuffle all the four time series data for ntest times
    # data_all = np.array([x, y, x_future, y_future])
    # data_shuffled_all = shuffle_x(data_all, sstmethod, ntest)
    # x_shuffled_all, x_future_shuffled_all = data_shuffled_all[0], data_shuffled_all[2]
    # y_shuffled_all, y_future_shuffled_all = data_shuffled_all[1], data_shuffled_all[3]

    # # Calculate the CCM score of each pair of (x_shuffled and y)
    # rho_shuffled_all = []
    # for i in range(ntest):
    #     # Get shuffled data
    #     x, x_future = x_shuffled_all[i], x_future_shuffled_all[i]
    #     y, y_future = y_shuffled_all[i], y_future_shuffled_all[i]
    #     # Calculate the corresponding CCM skill
    #     _, rho_shuffled = ccm(x, y, x_future, y_future, nemb, tau, nn, scoremethod, filtered)
    #     rho_shuffled_all.append(rho_shuffled)

    # # Calculate 95% and 5% percentiles
    # upper = np.percentile(rho_shuffled_all, 95)
    # lower = np.percentile(rho_shuffled_all, 5)

    # # Return results:
    # if rho > upper or rho < lower:
    #     return True, rho, upper, lower
    # else:
    #     return False, rho, upper, lower


def conductSST_withlag(xdata, ydata, lag, nx=25, ny=25, ntest=100, approach='kde_c', atomCheck=True, returnTrue=False):
    """Conduct the statistical significance test on the mutual information of two variables (X, Y) based on different lags.

    The shuffling procedure will be conducted on X.
    The calculated mi is significant if mi > mean(mi_shuffled_all) + 3*std(mi_shuffled_all)
    Input:
    xdata      -- the data series of the variable to be shuffled [ndarray with shape (nsamples)]
    ydata      -- the data series of the another variable [ndarray with shape (nsamples)]
    lag        -- the maximum lag
    nx         -- the number of bins in x dimension
    ny         -- the number of bins in y dimension
    ntest      -- the number of the shuffle time [int]
    approach   --  the approach used for computing PDF [str]
    atomCheck  -- indicating whether the atom-at-zero effect is requried [bool]
    returnTrue --  indicating whether the true mutual information is returned if the significant test fails [bool]
    Output:
    lagset     -- the set of the used lags [ndarray with shape (2*lag+1)]
    result_set -- the set of whether the mutual information is significant for different lags [ndarray with shape (2*lag+1)]
    mi_set     -- the set of the mutual information for different lags [ndarray with shape (2*lag+1)]
    """
    # Get the lag set
    lagset = np.arange(-lag, lag+1, 1)
    lagsetsize = lagset.size

    # Initialize result_set and mi_set
    result_set = np.ones(lagsetsize, dtype=bool)
    mi_set     = np.ones(lagsetsize, dtype='float64')

    # Conduct sst for each pair of x[t] and y[t+lag]
    for i in lagset:
        l = lagset[i]
        if l > 0:
            x, y = xdata[:-l], ydata[l:]
        elif l < 0:
            x, y = xdata[-l:], ydata[:l]
        else:
            x, y = xdata, ydata

        result, mi, _, _ = conductSST(xdata=x, ydata=y, nx=nx, ny=ny, ntest=ntest,
                                      approach=approach, atomCheck=atomCheck, returnTrue=returnTrue)
        result_set[i] = result
        mi_set[i]     = mi

    return lagset, result_set, mi_set


def conductSST(xdata, ydata, nx=25, ny=25, ntest=100, approach='kde_c', atomCheck=True, returnTrue=False):
    """Conduct the statistical significance test on the mutual information of two variables (X, Y).

    The shuffling procedure will be conducted on X.
    The calculated mi is significant if mi > 95% of the surrogates
    Input:
    xdata      -- the data series of the variable to be shuffled [ndarray with shape (nsamples)]
    ydata      -- the data series of the another variable [ndarray with shape (nsamples)]
    nx         -- the number of bins in x dimension
    ny         -- the number of bins in y dimension
    ntest      -- the number of the shuffle time [int]
    approach   --  the approach used for computing PDF [str]
    atomCheck  -- indicating whether the atom-at-zero effect is requried [bool]
    returnTrue --  indicating whether the true mutual information is returned if the significant test fails [bool]
    Output:
    result -- whether the mutual information is significant [boolean]
    mi     -- the mutual information
    lower  -- the lower threshold of the mi based on the surrogated data sets (5%) [float]
    upper  -- the upper threshold of the mi based on the surrogated data sets (95%) [float]
    """
    # Check whether xdata and ydata have the same length and are 1D
    if xdata.ndim != 1 or ydata.ndim != 1:
        raise Exception('The dimension of the input data should be 1D!')
    if xdata.shape != ydata.shape:
        raise Exception('The length of the two data sets are not the same!')

    # Calculate the mutual information of them
    data = np.array([xdata, ydata]).T
    pdfsolver = pdfComputer(ndim=2, approach=approach, atomCheck=atomCheck, bandwidth='silverman')
    _, pdf, _ = pdfsolver.computePDF(data, [nx, ny])
    mi = info(pdf, base=2).ixy

    # Shuffle xdata for ntest times, and
    xdata_shuffled_all = map(lambda i: np.random.permutation(xdata), range(ntest))

    # calculate the mutual information of each pair of (xdata_shuffled and ydata)
    mi_shuffled_all = []
    for i in range(ntest):
        # Get shuffled data
        xdata_shuffled = xdata_shuffled_all[i]
        # Calculate its pdf and mi
        data = np.array([xdata_shuffled, ydata]).T
        # pdfsolver = pdfComputer(ndim=2, approach='kde', bandwidth='silverman', kernel='epanechnikov')
        _, pdf, _ = pdfsolver.computePDF(data, [nx, ny])
        mi_shuffled = info(pdf, base=2).ixy
        mi_shuffled_all.append(mi_shuffled)

    # Calculate 95% and 5% percentiles
    upper = np.percentile(mi_shuffled_all, 95)
    lower = np.percentile(mi_shuffled_all, 5)

    # Return results:
    if mi > upper:
        return True, mi, upper, lower
    else:
        if returnTrue:  # Return true value of mutual information
            return False, mi, upper, lower
        else:
            return False, 0, upper, lower

    # # Conduct the statistical significance test
    # mean_mi_shuffled = np.mean(mi_shuffled_all)
    # std_mi_shuffled  = np.std(mi_shuffled_all)

    # # Return results
    # if mi > mean_mi_shuffled+3*std_mi_shuffled:
    #     return True, mi, mean_mi_shuffled, std_mi_shuffled
    # else:
    #     if returnTrue:  # Return true value of mutual information
    #         return False, mi, mean_mi_shuffled, std_mi_shuffled
    #     else:           # Return zero value for mutual information
    #         return False, 0., mean_mi_shuffled, std_mi_shuffled
    # # print mean_mi_shuffled, std_mi_shuffled, mi
