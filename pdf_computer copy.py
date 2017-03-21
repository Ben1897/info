# A class for computing multi-dimensional PDF by using scikit-learn kernel density function
#
# For the fixed bin method, the numpy histogram, histogram2d and histogramdd are used.
# For KDE, the scikit-learn's KernelDensity function is used.
#
# Note:
# A better illustration of KDE implementation in python can be refered to
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
#
# @Author: Peishi Jiang <Ben1897>
# @Date:   2017-02-13T11:25:34-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Ben1897
# @Last modified time: 2017-03-05T14:41:14-06:00

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
import time

# data types
float64 = 'float64'
boolean = bool


class pdfComputer(object):

    allowedApproach = ['fixedBin', 'kde']
    allowedBandwidthMethod = ['silverman', 'crossvalidation']

    def __init__(self, ndim, approach='kde', **kwargs):
        '''
        ndim: the number of dimension [int]
        approach: the selected approach for computing PDF [string]
        **kwargs: other key-value parameters;
                  for 'kde', it includes:
                  kernel   -- the kernel type [string]
                  bandwith -- the band with of the kernel [string or float]
                  other parameters used in KernelDensity;
                  for 'fixedBin', it includes:
                  all the parameters used in histogram, histogram2d and histogramdd
                  (usually, only number of bins is enough)
        '''
        self.ndim = ndim

        # Check whether the selected approach is within the allowed approach
        if approach not in self.allowedApproach:
            raise Exception('Unknown pdf estimation approach %s!' % approach)
        self.approach = approach

        # Check whether the provided parameters fulfill the requirements of the
        # selected approach.
        self.__checkApproachPara(**kwargs)

    def __checkApproachPara(self, **kwargs):
        '''
        Check whether the provided parameters fulfill the requirements of the
        selected approach.
        Note: most of the parameters checks will be taken care of by the pdf estimation
        function used. Currently, we will only consider how to calculate the bandwith
        for a specific kernel type.
        Output: NoneType.
        '''
        self.approachPara = {}
        if self.approach == 'fixedBin':  # fixed bin method
            if 'bins' in kwargs:
                self.approachPara['bins'] = kwargs['bins']
        elif self.approach == 'kde':  # KDE
            if 'bandwidth' in kwargs:
                self.approachPara['bandwidth'] = kwargs['bandwidth']
            if 'kernel' in kwargs:
                self.approachPara['kernel'] = kwargs['kernel']
            if 'atol' in kwargs:
                self.approachPara['atol'] = kwargs['atol']
            if 'rtol' in kwargs:
                self.approachPara['rtol'] = kwargs['rtol']

    def __checkAtomAtZero(self, data):
        '''
        Check the atom at zero effects of each dimension in data.
        Input:
            data -- the data [numpy array with shape (npoints, ndim)]
        Output (self):
            atom -- indicate whether one dimension needs consider atom-at-zero effect [ndarray with shape (ndim,), bool]
            kx   -- the proportion of nonzero values in each dimension [ndarray with shape (ndim,), float64]
        '''
        npts, ndim = data.shape
        thres      = .9          # The threshold to determine whether an atom-at-zero effect works

        # Compute kx
        kx = map(lambda i: np.sum(data[:, i] != 0)/float(npts), range(ndim))
        kx = np.array(kx, dtype=float64)

        # Compute atom
        # atom is set to zero if the corresponding kx is less than thres. Also,
        # If kx is larger than thres, kx is set to 1.
        atom = np.zeros(ndim, dtype=boolean)
        atom[kx < thres] = True
        kx[kx > thres]   = 1.

        self.kx   = kx
        self.atom = atom

    def computePDF(self, data, nbins):
        '''
        Compute the PDF based on selected approach.
        Input:
            data  -- the data [numpy array with shape (npoints, ndim)]
            nbins -- the number of bins in each dimension [list]
        Output:
            time   -- [float]
            pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
            coords -- [a numpy array with shape (ndim,)]
        '''
        # Check whether the row dimension of data equals to ndim
        if data.shape[1] != self.ndim:
            raise Exception('The dimension of the provided data does not equal to ndim!')

        # Check whether the length of n equals to ndim:
        if len(nbins) != self.ndim:
            raise Exception('The dimension of the provided bins does not equal to ndim!')

        # Check the atom-at-zero effect
        self.__checkAtomAtZero(data)

        # Computer the PDF
        if self.approach == 'kde':
            return self.computePDFKDE(data, nbins)
        elif self.approach == 'fixedBin':
            return self.computePDFFixedBin(data, nbins)

    def computePDFKDE(self, data, nbins):
        '''
        Compute the PDF based on KDE.
        Input:
        data  -- the data [numpy array with shape (npoints, ndim)]
        nbins -- the number of bins in each dimension [list]
        Output:
        time   -- [float]
        pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
        coords -- [a numpy array with shape (ndim,)]
        '''
        para = self.approachPara.copy()

        t0 = time.clock()
        # Get the band width value given the band width type
        if isinstance(para['bandwidth'], str):
            para['bandwidth'] = self.computeBandWidth(data, para['bandwidth'])

        # Compute the sampled points based on the band width
        # bd = para['bandwidth']
        if self.ndim == 1:
            nx = nbins[0]
            xarray = data[:, 0]
            xmin, xmax = xarray.min(), xarray.max()
            coords = np.linspace(xmin, xmax, nx)
            samples = coords[:, np.newaxis]
        elif self.ndim == 2:
            nx, ny = nbins[0], nbins[1]
            xarray, yarray = data[:, 0], data[:, 1]
            xmin, xmax = xarray.min(), xarray.max()
            ymin, ymax = yarray.min(), yarray.max()
            xcoords, ycoords = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
            xpts, ypts = np.meshgrid(xcoords, ycoords, indexing='ij')
            samples = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size)]).T
            coords = np.array([xcoords, ycoords])
        elif self.ndim == 3:
            nx, ny, nz = nbins[0], nbins[1], nbins[2]
            xarray, yarray, zarray = data[:, 0], data[:, 1], data[:, 2]
            xmin, xmax = xarray.min(), xarray.max()
            ymin, ymax = yarray.min(), yarray.max()
            zmin, zmax = zarray.min(), zarray.max()
            xcoords, ycoords, zcoords = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), np.linspace(zmin, zmax, nz)
            xpts, ypts, zpts = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
            samples = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size), zpts.reshape(zpts.size)]).T
            coords = np.array([xcoords, ycoords, zcoords])

        # Compute the PDF
        kde_skl = KernelDensity(**para)
        kde_skl.fit(data)
        log_pdf = kde_skl.score_samples(samples)
        pdf = np.exp(log_pdf)

        # Normalize the PDF and reorganize
        pdf = pdf / np.sum(pdf)
        if self.ndim == 2:
            # pdf = pdf.reshape(ycoords.size, xcoords.size)
            pdf = pdf.reshape(xcoords.size, ycoords.size)
        elif self.ndim == 3:
            # pdf = pdf.reshape(zcoords.size, ycoords.size, xcoords.size)
            pdf = pdf.reshape(xcoords.size, ycoords.size, zcoords.size)

        return time.clock() - t0, pdf, coords

    def computePDFFixedBin(self, data, nbins):
        '''
        Compute the PDF based on the fixed bin method.
        Input:
        data  -- the data [numpy array with shape (npoints, ndim)]
        nbins -- the number of bins in each dimension [list]
        Output:
        time   -- [float]
        pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
        coords -- [a numpy array with shape (ndim,)]
        '''
        para = self.approachPara.copy()
        para['bins'] = nbins

        # Compute the PDF
        t0 = time.clock()
        if self.ndim == 1:
            para['bins'] = nbins[0]
            pdf, edges = np.histogram(data, density=True, **para)
            coords = np.array([(edges[i+1]+edges[i])/2 for i in range(edges.size-1)])
        elif self.ndim == 2:
            x_array, y_array = data[:, 0], data[:, 1]
            pdf, xedges, yedges = np.histogram2d(x_array, y_array, normed=True, **para)
            coords = np.array([[(xedges[i+1]+xedges[i])/2 for i in range(xedges.size-1)],
                               [(yedges[i+1]+yedges[i])/2 for i in range(yedges.size-1)]])
        elif self.ndim == 3:
            pdf, edges = np.histogramdd(data, normed=True, **para)
            xedges, yedges, zedges = edges[0], edges[1], edges[2]
            coords = np.array([[(xedges[i+1]+xedges[i])/2 for i in range(xedges.size-1)],
                               [(yedges[i+1]+yedges[i])/2 for i in range(yedges.size-1)],
                               [(zedges[i+1]+zedges[i])/2 for i in range(zedges.size-1)]])

        # Normalize the PDF
        pdf = pdf / np.sum(pdf)

        return time.clock() - t0, pdf, coords

    def computeBandWidth(self, data, bandwidthType):
        '''
        Compute the band width given the type.
        Input:
        bandwidthType -- the type of band width [string]
        Output: [float]
        '''
        if bandwidthType == 'silverman':
            hlist = []
            for i in range(self.ndim):
                xarray = data[:, i]
                h = self.silverman(xarray)
                hlist.append(h)
            return np.min(hlist)
        elif bandwidthType == 'crossvalidation':
            return self.crossValidation(data)

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
