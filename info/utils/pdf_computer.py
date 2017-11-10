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

from .kdetoolkit import kde_c, kde_cuda, kde_cuda_general, kde_sklearn, kde_scipy


# data types
float64 = 'float64'
boolean = bool
integer = int


class pdf_computer(object):

    allowedApproach = ['kde_sklearn', 'kde_scipy', 'kde_cuda', 'kde_cuda_general', 'kde_c']
    allowedBandwidthMethod = ['silverman', 'scott']
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
        if approach == 'kde_sklearn':
            self.estimator = kde_sklearn
        elif approach == 'kde_c':
            self.estimator = kde_c
        elif approach == 'kde_cuda':
            self.estimator = kde_cuda
        elif approach == 'kde_cuda_general':
            self.estimator = kde_cuda_general
        elif approach == 'kde_scipy':
            self.estimator = kde_scipy

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
        approach  = self.approach

        # Get the number of data points
        npts, ndim = data.shape

        # Compute the bandwidth
        bd = self.computeBandWidth(data)

        # Estimate PDF
        if approach == 'kde_cuda_general':
            if ndim > 1:
                bdinv = np.linalg.inv(bd)
                bddet = np.linalg.det(bd)
            else:
                bdinv, bddet = 1./bd, bd
            pdf, t = estimator(ndim=ndim, kernel=kernel, bdinv=bdinv, bddet=bddet, Nt=npts,
                               No=npts, coordo=data, coordt=data, dtype=float64, rtime=True)
        else:
            pdf, t = estimator(ndim=ndim, kernel=kernel, bd=bd, Nt=npts, No=npts,
                               coordo=data, coordt=data, dtype=float64, rtime=True)

        # Normalize the PDF
        if normalized:
            pdf = pdf / np.sum(pdf)

        return t, pdf

    def computeBandWidth(self, data):
        '''
        Compute the band width given the type.
        Input:
        bandwidthType -- the type of band width [string]
        Output: [ndarray with shape(ndim,)]
        '''
        bandwidth  = self.bandwidth
        approach   = self.approach
        npts, ndim = data.shape

        # Compute the bandwidth
        if bandwidth == 'silverman':
            h = self.silverman(npts, ndim)
        elif bandwidth == 'scott':
            h = self.scott(npts, ndim)

        # Compute the covariance of data
        # Notice that in the general case, we are computing the squared value of the bandwidth
        if approach == 'kde_cuda_general':
            # covariance based
            bd = h**2 * np.cov(data.T)
        else:
            # std based
            if ndim > 1:
                cov  = np.cov(data.T)
                stds = np.sqrt(np.diagonal(cov))
            else:
                stds = data.std()
            bd = h * stds

        return bd

        # hlist = np.zeros(ndim)
        # for i in range(ndim):
        #     xarray   = data[:, i]
        #     if bandwidthType == 'silverman':
        #         h = self.silverman(xarray, ndim)
        #     elif bandwidthType == 'crossvalidation':
        #         h = self.crossValidation(data)
        #     hlist[i] = h
        # return hlist

    def silverman(self, npts, ndim):
        '''
        Compute the band width by using the Silverman's method.
        Input:
        npts -- the number of data points
        ndim -- the number of dimensions
        Output: [float]
        '''
        # Ref:
        # Eq.(6.43) in Scott's Multivariate Density Estimation: Theory, Practice, and Visualization, Second Edition (2015)
        # h = (4./(ndim+2))**(1./(ndim+4)) * std * n ** (-1./(ndim+4))
        h  = (4./(ndim+2))**(1./(ndim+4)) * npts ** (-1./(ndim+4))

        return h

    def scott(self, npts, ndim):
        '''
        Compute the band width by using the Silverman's method.
        Input:
        npts -- the number of data points
        ndim -- the number of dimensions
        Output: [float]
        '''
        # Ref:
        # Eq.(6.44) in Scott's Multivariate Density Estimation: Theory, Practice, and Visualization, Second Edition (2015)
        # h = (4./(ndim+2))**(1./(ndim+4)) * std * n ** (-1./(ndim+4))
        h  = npts ** (-1./(ndim+4))

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
