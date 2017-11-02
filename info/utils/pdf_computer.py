"""
A class for computing multi-dimensional PDF by using scikit-learn kernel density function.

For the fixed bin method, the numpy histogram, histogram2d and histogramdd are used.
For KDE, the scikit-learn's KernelDensity function is used.

Note:
A better illustration of KDE implementation in python can be refered to
https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

@Author: Peishi Jiang <Ben1897>
@Email:  shixijps@gmail.com

class pdfComputer()
    __init__()
    computePDF()
    computeBandWidth()
    silverman()
    crossvalidation()

"""


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity

from .kdetoolkit import kde_c, kde_cuda, kde_sklearn


# data types
float64 = 'float64'
boolean = bool
integer = int


class pdf_computer(object):

    allowedApproach = ['fixedBin', 'kde', 'kde_cuda', 'kde_c']
    allowedBandwidthMethod = ['silverman', 'crossvalidation']
    allowedKernels = ['gaussian', 'epanechnikov']

    def __init__(self, approach='kde_c', bandwidth='silverman', kernel='gaussian'):
        '''
        approach    -- the code for computing PDF by using KDE
        kernel      -- the kernel type [string]
        bandwith    -- the band with of the kernel [string or float]
        '''

        # Check
        if approach not in self.allowedApproach:
            raise Exception('Unknown pdf estimation approach %s!' % approach)
        if bandwidth not in self.allowedBandwidthMethod:
            raise Exception('Unknown KDE bandwidth type %s!' % bandwidth)
        self.approach = approach
        self.bandwidth = bandwidth
        self.kernel = kernel

        # Assign the estimator
        if approach == 'kde':
            self.estimator = kde_sklearn
        elif approach == 'kde_c':
            self.estimator = kde_c
        elif approach == 'kde_cuda':
            self.estimator = kde_cuda

    def computePDF(self, data, normalized=False):
        '''
        Compute the PDF based on selected approach.
        It should be noted that the mixed distribution case (atom-at-zero effect)
        only considers when the atom occurs at the corner of the whole defined region.
        Input:
            data -- the data [numpy array with shape (npoints, ndim)]
        Output:
            t    -- [float]
            pdf  -- [numpy array with shape (npoints)]
        '''
        estimator = self.estimator
        kernel    = self.kernel
        bandwidth = self.bandwidth

        # Get the number of data points
        npts, ndim = data.shape

        # Compute the bandwidth
        bd = self.computeBandWidth(data, bandwidth)

        # Estimate PDF
        pdf, t = estimator(ndim=ndim, kernel=kernel, bd=bd, Nt=npts, No=npts,
                           coordo=data, coordt=data, dtype=float64, rtime=True)

        # Normalize the PDF
        if normalized:
            pdf = pdf / np.sum(pdf)

        return t, pdf

    def computeBandWidth(self, data, bandwidthType):
        '''
        Compute the band width given the type.
        Input:
        bandwidthType -- the type of band width [string]
        Output: [ndarray with shape(ndim,)]
        '''
        npts, ndim = data.shape

        hlist = np.zeros(ndim)
        for i in range(ndim):
            xarray   = data[:, i]
            if bandwidthType == 'silverman':
                h = self.silverman(xarray)
            elif bandwidthType == 'crossvalidation':
                h = self.crossValidation(data)
            hlist[i] = h
        return hlist

    def silverman(self, xarray):
        '''
        Compute the band width by using the Silverman's method using for 1D.
        Input:
        xarray -- a numpy array
        Output: [float]
        '''
        std = np.std(xarray)
        n   = xarray.size

        if n == 1:
            raise Exception('There is only one value in xarray!')

        h   = 1.06 * std * n ** (-.2)

        return h

    def crossValidation(self, data):
        '''
        Compute the band width by using the cross validation method.
        Input:
        data -- a numpy array
        Output: [float]
        '''
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(.1, 1.0, 30)}, cv=20)
        grid.fit(data)
        return grid.best_params_['bandwidth']
