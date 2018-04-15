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

from ..core.info import info, computeMI, computeCMI, computeMIKNN, computeCMIKNN
from .others import reorganize_data, dropna

sstmethod_set = ['traditional', 'segments', 'seasonal']
kde_approaches = ['kde_c', 'kde_cuda', 'kde_cuda_general']
knn_approaches = ['knn_cuda', 'knn_scipy', 'knn_sklearn', 'knn']


def shuffle(data, shuffle_ind=[0], sstmethod='traditional'):
    """Shuffle data based on the shuffling method.

    Input:
    data      -- the data series of the variable to be shuffled [ndarray with shape (npts, ndim)]
    shuffle_ind -- the list of indices for shuffle test [list]
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
            # data_shuffled[:, i] = np.random.permutation(data[:, i])
            if i in shuffle_ind:
                data_shuffled[:, i] = np.random.permutation(data[:, i])
            else:
                data_shuffled[:, i] = data[:, i]
    elif sstmethod == 'segments':
        raise Exception('Not ready for the seasonal surrogates!')
        # ind_set = np.random.randint(low=0, high=nsamples, size=n)
        # ind_p = map(lambda i: np.roll(ind, -i), ind_set)
    elif sstmethod == 'seasonal':
        raise Exception('Not ready for the seasonal surrogates!')
    else:
        raise Exception('Unknown method %s' % sstmethod)

    return data_shuffled


def independence(node1, node2, data, shuffle_ind=[0], ntest=100, alpha=0.05,  approach='kde_cuda_general',
                 bandwidth='silverman', sstmethod='traditional', kernel='gaussian',  # Parameters for KDE
                 k=5,                                                                # Parameters for KNN
                 base=2., xyindex=[1], returnTrue=False):
    """
    Conduct the independence test based on the mutual information of two variables from data.

    Inputs:
    node1      -- the first node of interest with format (index, tau) [tuple]
    node2      -- the second node of interest with format (index, tau) [tuple]
    data       -- the data points [numpy array with shape (npoints, ndim)]
    shuffle_ind -- the list of indices for shuffle test [list]
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

    # Drop the rows with nan values
    data12n = dropna(data12)

    # Calculate the mutual information of them
    if approach in kde_approaches:
        mi = info(case=2, data=data12n, approach=approach, kernel=kernel, bandwidth=bandwidth,
                  base=base, xyindex=xyindex, conditioned=False).ixy
        # mi = computeMI(data=data12n, approach=approach, xyindex=xyindex, kernel=kernel, base=base)
    elif approach in knn_approaches:
        mi = computeMIKNN(data=data12n, k=k, xyindex=xyindex)

    # Calculate the mutual information of each pair of (xdata_shuffled and ydata)
    mi_shuffled_all = []
    for i in range(ntest):
        # Get shuffled data
        data12_shuffled = shuffle(data12, shuffle_ind, sstmethod=sstmethod)

        # Drop the rows with nan values
        data12n_shuffled = dropna(data12_shuffled)

       # Calculate the corresponding mi
        if approach in kde_approaches:
            mi_shuffled = info(case=2, data=data12n_shuffled, approach=approach, kernel=kernel, bandwidth=bandwidth,
                               base=base, xyindex=xyindex, conditioned=False).ixy
            # mi_shuffled = computeMI(data=data12n_shuffled, approach=approach, xyindex=xyindex, kernel=kernel, base=base)
        elif approach in knn_approaches:
            mi_shuffled = computeMIKNN(data=data12n_shuffled, k=k, xyindex=xyindex)

        mi_shuffled_all.append(mi_shuffled)

    # Calculate 95% and 5% percentiles
    upper = np.percentile(mi_shuffled_all, int(100*(1-alpha)))
    lower = np.percentile(mi_shuffled_all, int(100*alpha))

    # Return results:
    if mi > upper:
    # if mi > upper or mi < lower:
        if returnTrue:
            return False, mi, upper, lower
        else:
            return False
    else:
        if returnTrue:  # Return true value of mutual information
            return True, mi, upper, lower
        else:
            return True

def conditionalIndependence(node1, node2, conditionset, data, shuffle_ind=[0], ntest=100, alpha=0.05,  approach='kde_cuda_general',
                            bandwidth='silverman', sstmethod='traditional', kernel='gaussian',  # Parameters for KDE
                            k=5,                                                                # Parameters for KNN
                            base=2., xyindex=[1,2], returnTrue=False):
    """
    Conduct the conditional independence test based on the conditional mutual information of two variables from data.

    Inputs:
    node1        -- the first node of interest with format (index, tau) [tuple]
    node2        -- the second node of interest with format (index, tau) [tuple]
    conditionset -- the condition set with format [(index, tau)] [list of tuples]
    data         -- the data points [numpy array with shape (npoints, ndim)]
    shuffle_ind -- the list of indices for shuffle test [list]
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

    # Drop the rows with nan values
    data12condn = dropna(data12cond)

    # Calculate the conditional mutual information of them
    if approach in kde_approaches:
        cmi = info(case=2, data=data12condn, approach=approach, kernel=kernel, bandwidth=bandwidth,
                   base=base, xyindex=xyindex, conditioned=True).ixy_w
        # cmi = computeCMI(data=data12condn, approach=approach, kernel=kernel, base=base)
    elif approach in knn_approaches:
        cmi = computeCMIKNN(data=data12condn, k=k, xyindex=xyindex)

    # Calculate the conditional mutual information of each pair of (xdata_shuffled and ydata)
    cmi_shuffled_all = np.zeros(ntest)
    for i in range(ntest):
        # Get shuffled data
        data12cond_shuffled = shuffle(data12cond, shuffle_ind, sstmethod=sstmethod)

        # Drop the rows with nan values
        data12condn_shuffled = dropna(data12cond_shuffled)

       # Calculate the corresponding cmi
        if approach in kde_approaches:
            cmi_shuffled = info(case=2, data=data12condn_shuffled, approach=approach, kernel=kernel, bandwidth=bandwidth,
                                base=base, xyindex=xyindex, conditioned=True).ixy_w
            # cmi_shuffled = computeCMI(data=data12condn_shuffled, approach=approach, kernel=kernel, base=base)
        elif approach in knn_approaches:
            cmi_shuffled = computeCMIKNN(data=data12condn_shuffled, k=k, xyindex=xyindex)

        cmi_shuffled_all[i] = cmi_shuffled

    # Calculate 95% and 5% percentiles
    upper = np.percentile(cmi_shuffled_all, int(100*(1-alpha)))
    lower = np.percentile(cmi_shuffled_all, int(100*alpha))

    # print cmi_shuffled_all.max(), cmi_shuffled_all.min()
    # Return results:
    if cmi > upper:
    # if cmi > upper or cmi < lower:
        if returnTrue:
            return False, cmi, upper, lower
        else:
            return False
    else:
        if returnTrue:  # Return true value of mutual information
            return True, cmi, upper, lower
        else:
            return True
