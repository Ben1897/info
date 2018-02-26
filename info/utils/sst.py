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

from ..core.info import info
from .others import reorganize_data

sstmethod_set = ['traditional', 'segments', 'seasonal']


def shuffle(data, sstmethod='traditional'):
    """Shuffle data based on the shuffling method.

    Input:
    data      -- the data series of the variable to be shuffled [ndarray with shape (npts, ndim)]
    sstmethod -- the method used for generating the surrogate data [str]

    """
    npts, ndim = data.shape

    data_shuffled = np.zeros([npts, ndim])

    # Create the original indices
    ind = np.arange(0, npts, dtype=int)

    # Get the indices of all the permutations
    if sstmethod == 'traditional':
        # permutation = lambda i: np.random.permutation(ind)
        # ind_p = np.array(map(permutation, range(ndim))).T
        for i in range(ndim):
            data_shuffled[:, i] = np.random.permutation(data[:, i])
            # if i == 0:
            #     data_shuffled[:, i] = np.random.permutation(data[:, i])
            # else:
            #     data_shuffled[:, i] = data[:, i]
    elif sstmethod == 'segments':
        raise Exception('Not ready for the seasonal surrogates!')
        # ind_set = np.random.randint(low=0, high=nsamples, size=n)
        # ind_p = map(lambda i: np.roll(ind, -i), ind_set)
    elif sstmethod == 'seasonal':
        raise Exception('Not ready for the seasonal surrogates!')
    else:
        raise Exception('Unknown method %s' % sstmethod)

    return data_shuffled


def independence(node1, node2, data, ntest=100, sstmethod='traditional', alpha=0.05, kernel='gaussian', approach='kde_cuda_general', base=2., returnTrue=False):
    """
    Conduct the independence test based on the mutual information of two variables from data.

    Inputs:
    node1      -- the first node of interest with format (index, tau) [tuple]
    node2      -- the second node of interest with format (index, tau) [tuple]
    data       -- the data points [numpy array with shape (npoints, ndim)]
    ntest      -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]

    """
    # Get data for node1 and node2
    data12 = reorganize_data(data, [node1, node2])

    # Calculate the mutual information of them
    mi = info(case=2, data=data12, approach=approach, kernel=kernel,
              base=base, conditioned=False).ixy

    # Calculate the mutual information of each pair of (xdata_shuffled and ydata)
    mi_shuffled_all = []
    for i in range(ntest):
        # Get shuffled data
        data12_shuffled = shuffle(data12, sstmethod=sstmethod)
        # Calculate the corresponding mi
        mi_shuffled = info(case=2, data=data12_shuffled, approach=approach,
                           kernel=kernel, base=base, conditioned=False).ixy
        mi_shuffled_all.append(mi_shuffled)

    # Calculate 95% and 5% percentiles
    upper = np.percentile(mi_shuffled_all, int(100*(1-alpha)))
    lower = np.percentile(mi_shuffled_all, int(100*alpha))

    # Return results:
    if mi > upper:
        if returnTrue:
            return False, mi, upper, lower
        else:
            return False
    else:
        if returnTrue:  # Return true value of mutual information
            return True, mi, upper, lower
        else:
            return True


def conditionalIndependence(node1, node2, conditionset, data, ntest=100, sstmethod='traditional', alpha=0.05, kernel='gaussian', approach='kde_cuda_general', base=2., returnTrue=False):
    """
    Conduct the conditional independence test based on the conditional mutual information of two variables from data.

    Inputs:
    node1        -- the first node of interest with format (index, tau) [tuple]
    node2        -- the second node of interest with format (index, tau) [tuple]
    conditionset -- the condition set with format [(index, tau)] [list of tuples]
    data         -- the data points [numpy array with shape (npoints, ndim)]
    ntest        -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]

   """
    # Get data for node1 and node2
    data12cond = reorganize_data(data, [node1, node2]+conditionset)

    # Calculate the conditional mutual information of them
    cmi = info(case=2, data=data12cond, approach=approach, kernel=kernel,
              base=base, conditioned=True).ixy_w

    # Calculate the conditional mutual information of each pair of (xdata_shuffled and ydata)
    cmi_shuffled_all = np.zeros(ntest)
    for i in range(ntest):
        # Get shuffled data
        data12cond_shuffled = shuffle(data12cond, sstmethod=sstmethod)
        # Calculate the corresponding cmi
        cmi_shuffled = info(case=2, data=data12cond_shuffled, approach=approach,
                            kernel=kernel, base=base, conditioned=True).ixy_w
        cmi_shuffled_all[i] = cmi_shuffled

    # Calculate 95% and 5% percentiles
    upper = np.percentile(cmi_shuffled_all, int(100*(1-alpha)))
    lower = np.percentile(cmi_shuffled_all, int(100*alpha))

    # print cmi_shuffled_all.max(), cmi_shuffled_all.min()
    # Return results:
    if cmi > upper:
        if returnTrue:
            return False, cmi, upper, lower
        else:
            return False
    else:
        if returnTrue:  # Return true value of mutual information
            return True, cmi, upper, lower
        else:
            return True
