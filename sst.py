# A function for conducting the statistical significance test on the mutual information
# of two variables.
#
# @Author: Peishi Jiang <Ben1897>
# @Date:   2017-02-17T10:47:00-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Ben1897
# @Last modified time: 2017-03-15T20:53:59-05:00


from info import info
import numpy as np
from pdf_computer import pdfComputer


def conductSST(xdata, ydata, nx=25, ny=25, ntest=100, approach='kde_c', atomCheck=True, returnTrue=False):
    '''
    Conduct the statistical significance test on the mutual information of
    two variables (X, Y).
    The shuffling procedure will be conducted on X.
    The calculated mi is significant if mi > mean(mi_shuffled_all) + 3*std(mi_shuffled_all)
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
    '''

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

    # Conduct the statistical significance test
    mean_mi_shuffled = np.mean(mi_shuffled_all)
    std_mi_shuffled  = np.std(mi_shuffled_all)

    # Return results
    if mi > mean_mi_shuffled+3*std_mi_shuffled:
        return True, mi, mean_mi_shuffled, std_mi_shuffled
    else:
        if returnTrue:  # Return true value of mutual information
            return False, mi, mean_mi_shuffled, std_mi_shuffled
        else:           # Return zero value for mutual information
            return False, 0., mean_mi_shuffled, std_mi_shuffled
    # print mean_mi_shuffled, std_mi_shuffled, mi
