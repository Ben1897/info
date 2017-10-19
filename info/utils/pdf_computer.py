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
    computePDFmd()
    computePDF1d()
    computePDF2d()
    computePDF3d()
    computeBandWidth()
    silverman()
    crossvalidation()
    __checkApproachPara()
    __checkAtomAtZero()
    __computeEdgeCoord()

"""


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
import time

from .kdetoolkit import kde_c, kde_cuda, kde_sklearn


# data types
float64 = 'float64'
boolean = bool
integer = int


class pdfComputer(object):

    allowedApproach = ['fixedBin', 'kde', 'kde_cuda', 'kde_c']
    allowedBandwidthMethod = ['silverman', 'crossvalidation']

    def __init__(self, ndim, approach='kde_cuda', **kwargs):
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

        # Assign the estimator
        if approach == 'kde':
            self.estimator = kde_sklearn
        elif approach == 'kde_c':
            self.estimator = kde_c
        elif approach == 'kde_cuda':
            self.estimator = kde_cuda

        # Check whether the provided parameters fulfill the requirements of the
        # selected approach.
        self.__checkApproachPara(**kwargs)

    def computePDF(self, data, nbins, limits=None, atomCheck=True):
        '''
        Compute the PDF based on selected approach.
        It should be noted that the mixed distribution case (atom-at-zero effect)
        only considers when the atom occurs at the corner of the whole defined region.
        Input:
            data      -- the data [numpy array with shape (npoints, ndim)]
            nbins     -- the number of bins in each dimension [list]
            limits    -- the min and max limits of each axis [list]
            atomCheck -- a boolean value indicate whether the atom-at-zero effect is considered [bool]
        Output:
            time   -- [float]
            pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
            coords -- [a numpy array with shape (ndim,)]
        '''
        # Get the ndim value in ndim is 'm'
        usePDFmd = False
        if self.ndim == 'm':
            usePDFmd = True
            self.ndim = data.shape[1]
        # Check whether the row dimension of data equals to ndim
        if data.shape[1] != self.ndim:
            raise Exception('The dimension of the provided data does not equal to ndim!')
        self.npts = data.shape[0]

        # Check whether the length of n equals to ndim:
        if len(nbins) != self.ndim:
            raise Exception('The dimension of the provided bins does not equal to ndim!')

        # Check whether the provides limits comply with the dimension
        if limits is not None and len(limits) != self.ndim:
            raise Exception('The dimension of the provided limits does not equal to ndim!')
        self.limits = limits

        # Check the atom-at-zero effect
        if atomCheck:
            self.__checkAtomAtZero(data)
        else:
            self.kset = np.ones(self.ndim, dtype=float64)
            self.atom = np.zeros(self.ndim, dtype=integer)

        # Compute the edges and coordinates of each dimension
        self.__computeEdgeCoord(data, nbins)

        # If all values are zeros, then return PDF matrix where only the origin is 1
        if self.kset.sum() == 0:
            t0 = time.time()
            if self.ndim == 1:
                pdf    = np.zeros(nbins[0])
                pdf[0] = 1
            elif self.ndim == 2:
                pdf       = np.zeros(nbins[0], nbins[1])
                pdf[0, 0] = 1
            elif self.ndim == 3:
                pdf          = np.zeros(nbins[0], nbins[1], nbins[2])
                pdf[0, 0, 0] = 1
            return time.time()-t0, pdf, self.coords

        # Computer the PDF
        if usePDFmd:  # if the user would like to use computePDFmd function
            return self.computePDFmd(data, nbins)
        if self.ndim == 1:
            return self.computePDF1d(data, nbins)
        elif self.ndim == 2:
            return self.computePDF2d(data, nbins)
        elif self.ndim == 3:
            return self.computePDF3d(data, nbins)

    def computePDFmd(self, data, nbins):
        '''
        Compute the multi-dimensional PDF based on KDE.
        Input:
            data  -- the data [numpy array with shape (npoints, ndim)]
            nbins -- the number of bins in each dimension [list]
        Output:
            time   -- [float]
            pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
            coords -- [a numpy array with shape (ndim,)]
        '''
        npts    = self.npts
        ndim    = self.ndim
        coords  = self.coords
        estimator = self.estimator
        para   = self.approachPara.copy()

        t0 = time.time()

        # initialize PDF set
        pdf = np.zeros(nbins, dtype=float64)

        # Calculate the number of estimated points
        Nt = reduce((lambda x, y: x*y), nbins)

        # Generate the coordinates where pdfs are estimated with shape (Nt, ndim )
        # print coords
        coord  = np.meshgrid(*coords, indexing='ij')
        coordt = np.array(coord).reshape(ndim, Nt).T

        # Get the band width value given the band width type
        if isinstance(para['bandwidth'], str):
            bd = self.computeBandWidth(data, para['bandwidth'])

        # Estimate PDF
        pdf = estimator(ndim, bd, Nt=Nt, No=npts, coordo=data, coordt=coordt, dtype=float64)
        # print pdf.nbytes, pdf[0], pdf[1]
        # print np.count_nonzero(~np.isnan(pdf))
        pdf = pdf.reshape(nbins)

        # Normalize pdf again in case there is slice error
        pdf = pdf/pdf.sum()

        return time.time() - t0, pdf, self.coords

    def computePDF1d(self, data, nbins):
        '''
        Compute the 1D PDF based on KDE.
        Input:
            data  -- the data [numpy array with shape (npoints, ndim)]
            nbins -- the number of bins in each dimension [list]
        Output:
            time   -- [float]
            pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
            coords -- [a numpy array with shape (ndim,)]
        '''
        # npts   = self.npts
        ndim   = self.ndim
        kx     = self.kset[0]
        atom   = self.atom
        coords = self.coords
        para   = self.approachPara.copy()

        estimator = self.estimator

        t0 = time.time()

        # initialize PDF set
        pdf = np.zeros(nbins[0])

        # Get the band width value given the band width type
        if isinstance(para['bandwidth'], str):
            bd = self.computeBandWidth(data, para['bandwidth'])

        # Compute PDF center
        if atom[0]:  # consider atom-at-zero effect
            # Get the nonzeros values
            coordo = data[data != 0]
            No = coordo.size
            # Get the values to be estimated
            coordt = coords[0][1:, np.newaxis]
            Nt     = coordt.size
            pdfcenter = estimator(ndim, bd, Nt, No, coordo, coordt, dtype=float64)
        else:        # no atom-at-zero effect
            # Get the sample values
            coordo = data
            No = coordo.size
            # Get the values to be estimated
            coordt = coords[0][:, np.newaxis]
            Nt     = coordt.size
            pdfcenter = estimator(ndim, bd, Nt, No, coordo, coordt, dtype=float64)

        # Combine pdfcenter to pdf and compute the origin if it is a atom-at-zero effect
        if atom[0]:
            pdf[0]  = 1 - kx
            pdf[1:] = kx*pdfcenter
        else:
            pdf = pdfcenter

        # Normalize pdf again in case there is slice error
        pdf = pdf/pdf.sum()

        return time.time() - t0, pdf, coords

    def computePDF2d(self, data, nbins):
        '''
        Compute the 2D PDF based on KDE.
        Input:
            data  -- the data [numpy array with shape (npoints, ndim)]
            nbins -- the number of bins in each dimension [list]
        Output:
            time   -- [float]
            pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
            coords -- [a numpy array with shape (ndim,)]
        '''
        npts    = self.npts
        ndim    = self.ndim
        kx,ky   = self.kset[0], self.kset[1]
        atom    = self.atom
        xcoords = self.coords[0]
        ycoords = self.coords[1]
        para    = self.approachPara.copy()
        nx, ny  = nbins[0], nbins[1]

        estimator = self.estimator

        t0 = time.time()

        # Get indices of x == 0, y == 0 and x == y == 0
        xdata, ydata = data[:,0], data[:,1]
        xmin, ymin   = xdata.min(), ydata.min()
        indx0 = np.all([xdata==xmin, ydata!=ymin], axis=0)   # indices for y-axis (not including y=0)
        indy0 = np.all([xdata!=xmin, ydata==ymin], axis=0)   # indices for x-axis (not including x=0)
        ind00 = np.all([xdata==xmin, ydata==ymin], axis=0)   # indices for the origin
        indn0 = np.all([xdata!=xmin, ydata!=ymin], axis=0)   # indices for x != xmin and y != ymin
        # print data.shape
        # print indx0.sum(), indy0.sum(), ind00.sum(), indn0.sum()

        # initialize PDF set
        pdf = np.zeros([nx, ny], dtype=float64)

        # Get the band width value given the band width type
        if isinstance(para['bandwidth'], str):
            bd = self.computeBandWidth(data, para['bandwidth'])

        # Compute PDF center, pdfxo and pdfyo
        if atom[0] == 0 and atom[1] == 0:    # no atom-at-zero effect
            # Compute PDF center and reshape it
            xpts, ypts = np.meshgrid(xcoords, ycoords, indexing='ij')
            coordt     = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size)]).T
            xcoordt    = xcoords[:]
            ycoordt    = ycoords[:]
            Nt         = coordt.shape[0]
            pdfcenter  = estimator(ndim, bd, Nt, npts, data, coordt, dtype=float64)
            pdfcenter  = pdfcenter.reshape(xcoordt.size, ycoordt.size)
        elif atom[0] == 1 and atom[1] == 1:  # consider atom-at-zero effect at both x and y
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            coordo  = data[indn0, :]
            xcoordt = xcoords[1:]
            ycoordt = ycoords[1:]
            # Compute PDF center and reshape it
            xpts, ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            coordt     = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size)]).T
            No, Nt     = coordo.shape[0], coordt.shape[0]
            pdfcenter  = estimator(ndim, bd, Nt, No, coordo, coordt, dtype=float64)
            pdfcenter  = pdfcenter.reshape(xcoordt.size, ycoordt.size)
            ### Compute pdfx0 and pdfy0
            x0data = xdata[indy0]
            y0data = ydata[indx0]
            pdfx0  = estimator(1, bd[0], nx-1, x0data.size, x0data, xcoordt[:,np.newaxis], dtype=float64)
            pdfy0  = estimator(1, bd[1], ny-1, y0data.size, y0data, ycoordt[:,np.newaxis], dtype=float64)
        elif atom[0] == 0 and atom[1] == 1:  # consider atom-at-zero effect at x (reduced to 1D)
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indx0], axis=0)  # the indices of all the values except y==0
            coordo  = data[ind, :]
            xcoordt = xcoords[:]
            ycoordt = ycoords[1:]
            # Compute PDF center and reshape it
            xpts, ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            coordt     = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size)]).T
            No, Nt     = coordo.shape[0], coordt.shape[0]
            pdfcenter  = estimator(ndim, bd, Nt, No, coordo, coordt, dtype=float64)
            pdfcenter  = pdfcenter.reshape(xcoordt.size, ycoordt.size)
            # Compute pdfx0
            ind    = np.any([ind00, indy0], axis=0)
            x0data = xdata[ind]
            pdfx0  = estimator(1, bd[0], nx, x0data.size, x0data, xcoordt[:,np.newaxis], dtype=float64)
        elif atom[0] == 1 and atom[1] == 0:  # consider atom-at-zero effect at y (reduced to 1D)
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indy0], axis=0)  # the indices of all the values except x==0
            coordo  = data[ind, :]
            xcoordt = xcoords[1:]
            ycoordt = ycoords[:]
            # Compute PDF center and reshape it
            xpts, ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            coordt     = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size)]).T
            No, Nt     = coordo.shape[0], coordt.shape[0]
            pdfcenter  = estimator(ndim, bd, Nt, No, coordo, coordt, dtype=float64)
            pdfcenter  = pdfcenter.reshape(xcoordt.size, ycoordt.size)
            # Compute pdfy0
            ind    = np.any([ind00, indx0], axis=0)
            y0data = ydata[ind]
            pdfy0  = estimator(1, bd[1], ny, y0data.size, y0data, ycoordt[:,np.newaxis], dtype=float64)

        # Combine pdfcenter, pdfxo and pdfyo to pdf
        # and compute the origin if it is a atom-at-zero effect
        if atom[0] == 0 and atom[1] == 0:    # no atom-at-zero effect
            pdf = pdfcenter
        elif atom[0] == 1 and atom[1] == 1:  # consider atom-at-zero effect at both x and y
            pdf[0, 0]   = (1-kx)*(1-ky)
            pdf[1:, 1:] = kx*ky*pdfcenter
            pdf[0, 1:]  = ky*(1-kx)*pdfy0
            pdf[1:, 0]  = kx*(1-ky)*pdfx0
        elif atom[0] == 0 and atom[1] == 1:  # consider atom-at-zero effect at x (ky=1)
            pdf[:, 1:] = ky*pdfcenter
            pdf[:, 0]  = (1-ky)*pdfx0
            # print kx, ky
            # print pdfx0, pdfcenter.max()
        elif atom[0] == 1 and atom[1] == 0:  # consider atom-at-zero effect at y (kx=1)
            pdf[1:, :] = kx*pdfcenter
            pdf[0, :]  = (1-kx)*pdfy0
            # print pdfy0.max(), pdfcenter.max(), ky, kx
            # print pdf[0,:].max(), pdf[1:, :].max()

        # Normalize pdf again in case there is slice error
        pdf = pdf/pdf.sum()

        return time.time() - t0, pdf, self.coords

    def computePDF3d(self, data, nbins):
        '''
        Compute the 3D PDF based on KDE.
        Input:
            data  -- the data [numpy array with shape (npoints, ndim)]
            nbins -- the number of bins in each dimension [list]
        Output:
            time   -- [float]
            pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
            coords -- [a numpy array with shape (ndim,)]
        '''
        # npts     = self.npts
        ndim     = self.ndim
        kx,ky,kz = self.kset[0], self.kset[1], self.kset[2]
        atom     = self.atom
        xcoords  = self.coords[0]
        ycoords  = self.coords[1]
        zcoords  = self.coords[2]
        para     = self.approachPara.copy()
        nx,ny,nz = nbins[0], nbins[1], nbins[2]

        estimator = self.estimator

        t0 = time.time()

        # Get indices of x, y and z axes, xy panel, yz panel and xz panel
        xdata, ydata, zdata    = data[:,0], data[:,1], data[:,2]
        xmin, ymin, zmin       = xdata.min(), ydata.min(), zdata.min()
        xydata, xzdata, yzdata = data[:,[0,1]], data[:,[0,2]], data[:,[1,2]]
        indx0  = np.all([xdata==xmin, ydata!=ymin, zdata!=zmin], axis=0)    # indices for yz panel, not including y z axes
        indy0  = np.all([xdata!=xmin, ydata==ymin, zdata!=zmin], axis=0)    # indices for xz panel, not including x z axes
        indz0  = np.all([xdata!=xmin, ydata!=ymin, zdata==zmin], axis=0)    # indices for xy panel, not including x y axes
        indxy0 = np.all([xdata==xmin, ydata==ymin, zdata!=zmin], axis=0)    # indices for zaxis, not including the origin
        indxz0 = np.all([xdata==xmin, ydata!=ymin, zdata==zmin], axis=0)    # indices for yaxis, not including the origin
        indyz0 = np.all([xdata!=xmin, ydata==ymin, zdata==zmin], axis=0)    # indices for xaxis, not including the origin
        ind00  = np.all([xdata==xmin, ydata==ymin, zdata==zmin], axis=0)    # indices for the origin
        indn0  = np.all([xdata!=xmin, ydata!=ymin, zdata!=zmin], axis=0)    # indices for x != xmin and y != ymin and z != zmin

        # initialize PDF set
        pdf = np.zeros([nx, ny, nz], dtype=float64)

        # Get the band width value given the band width type
        if isinstance(para['bandwidth'], str):
            bd = self.computeBandWidth(data, para['bandwidth'])

        # Get all the indices and compute the PDF except the pdfcenter
        if atom[0] == 0 and atom[1] == 0 and atom[2] == 0:    # no atom-at-zero effect
            coordo  = data[indn0, :]
            xcoordt = xcoords[:]
            ycoordt = ycoords[:]
            zcoordt = zcoords[:]
        elif atom[0] == 1 and atom[1] == 1 and atom[2] == 1:  # consider atom-at-zero effect at both x, y and z
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            coordo  = data[indn0, :]
            xcoordt = xcoords[1:]
            ycoordt = ycoords[1:]
            zcoordt = zcoords[1:]
            ### Compute pdfx0, pdfy0, pdfz0
            x0data, y0data, z0data = xdata[indyz0], ydata[indxz0], zdata[indxy0]
            pdfx0  = estimator(1, bd[0], nx-1, x0data.size, x0data, xcoordt[:,np.newaxis], dtype=float64)
            pdfy0  = estimator(1, bd[1], ny-1, y0data.size, y0data, ycoordt[:,np.newaxis], dtype=float64)
            pdfz0  = estimator(1, bd[2], nz-1, z0data.size, z0data, ycoordt[:,np.newaxis], dtype=float64)
            ### Compute pdfxy, pdfyz, pdfxz
            # Get the data for x-y, y-z and x-z panels
            xy0data, yz0data, xz0data = xydata[indz0,:], yzdata[indx0,:], xzdata[indy0,:]
            # Get all the locations to be estimated for x-y, y-z and x-z panels
            xy0xpts, xy0ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            xy0coordt        = np.array([xy0xpts.reshape(xy0xpts.size), xy0ypts.reshape(xy0ypts.size)]).T
            yz0xpts, yz0ypts = np.meshgrid(ycoordt, zcoordt, indexing='ij')
            yz0coordt        = np.array([yz0xpts.reshape(yz0xpts.size), yz0ypts.reshape(yz0ypts.size)]).T
            xz0xpts, xz0ypts = np.meshgrid(xcoordt, zcoordt, indexing='ij')
            xz0coordt        = np.array([xz0xpts.reshape(xz0xpts.size), xz0ypts.reshape(xz0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfxy  = estimator(2, bd[[0,1]], xcoordt.size*ycoordt.size, xy0data.shape[0], xy0data, xy0coordt, dtype=float64)
            pdfyz  = estimator(2, bd[[1,2]], ycoordt.size*zcoordt.size, yz0data.shape[0], yz0data, yz0coordt, dtype=float64)
            pdfxz  = estimator(2, bd[[0,2]], zcoordt.size*xcoordt.size, xz0data.shape[0], xz0data, xz0coordt, dtype=float64)
            pdfxy  = pdfxy.reshape(xcoordt.size, ycoordt.size)
            pdfyz  = pdfyz.reshape(ycoordt.size, zcoordt.size)
            pdfxz  = pdfxz.reshape(xcoordt.size, zcoordt.size)
            # Replace nan with zero
            pdfx0, pdfy0, pdfz0 = np.nan_to_num(pdfx0), np.nan_to_num(pdfy0), np.nan_to_num(pdfz0)
            pdfxy, pdfyz, pdfxz = np.nan_to_num(pdfxy), np.nan_to_num(pdfyz), np.nan_to_num(pdfxz)
        elif atom[0] == 0 and atom[1] == 1 and atom[2] == 1:  # consider atom-at-zero effect at both y and z
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indx0], axis=0)  # the indices of non-zero values
            coordo  = data[ind, :]
            xcoordt = xcoords[:]
            ycoordt = ycoords[1:]
            zcoordt = zcoords[1:]
            ### Compute pdfx0
            ind = np.any([indyz0, ind00], axis=0)
            x0data = xdata[ind]
            pdfx0  = estimator(1, bd[0], xcoordt.size, x0data.size, x0data, xcoordt[:,np.newaxis], dtype=float64)
            ### Compute pdfxy, pdfxz
            # Get the data for x-y and x-z panels
            ind1, ind2       = np.any([indz0,indxz0], axis=0), np.any([indy0,indxy0], axis=0)
            xy0data, xz0data = xydata[ind1,:], xzdata[ind2,:]
            # Get all the locations to be estimated for x-y and x-z panels
            xy0xpts, xy0ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            xy0coordt        = np.array([xy0xpts.reshape(xy0xpts.size), xy0ypts.reshape(xy0ypts.size)]).T
            xz0xpts, xz0ypts = np.meshgrid(xcoordt, zcoordt, indexing='ij')
            xz0coordt        = np.array([xz0xpts.reshape(xz0xpts.size), xz0ypts.reshape(xz0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfxy  = estimator(2, bd[[0,1]], xcoordt.size*ycoordt.size, xy0data.shape[0], xy0data, xy0coordt, dtype=float64)
            pdfxz  = estimator(2, bd[[0,2]], xcoordt.size*zcoordt.size, xz0data.shape[0], xz0data, xz0coordt, dtype=float64)
            pdfxy  = pdfxy.reshape(xcoordt.size, ycoordt.size)
            pdfxz  = pdfxz.reshape(xcoordt.size, zcoordt.size)
            # Replace nan with zero
            pdfx0 = np.nan_to_num(pdfx0)
            pdfxy, pdfxz = np.nan_to_num(pdfxy), np.nan_to_num(pdfxz)
        elif atom[0] == 1 and atom[1] == 0 and atom[2] == 1:  # consider atom-at-zero effect at both x and z
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indy0], axis=0)  # the indices of non-zero values
            coordo  = data[ind, :]
            xcoordt, ycoordt, zcoordt = xcoords[1:], ycoords[:], zcoords[1:]
            ### Compute pdfy0
            ind = np.any([indxz0, ind00], axis=0)
            y0data = ydata[ind]
            pdfy0  = estimator(1, bd[0], ycoordt.size, y0data.size, y0data, ycoordt[:,np.newaxis], dtype=float64)
            ### Compute pdfxy, pdfyz
            # Get the data for x-y and y-z panels
            ind1, ind2       = np.any([indz0,indyz0], axis=0), np.any([indx0,indxy0], axis=0)
            xy0data, yz0data = xydata[ind1,:], yzdata[ind2,:]
            # Get all the locations to be estimated for x-y and y-z panels
            xy0xpts, xy0ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            xy0coordt        = np.array([xy0xpts.reshape(xy0xpts.size), xy0ypts.reshape(xy0ypts.size)]).T
            yz0xpts, yz0ypts = np.meshgrid(ycoordt, zcoordt, indexing='ij')
            yz0coordt        = np.array([yz0xpts.reshape(yz0xpts.size), yz0ypts.reshape(yz0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfxy  = estimator(2, bd[[0,1]], xcoordt.size*ycoordt.size, xy0data.shape[0], xy0data, xy0coordt, dtype=float64)
            pdfyz  = estimator(2, bd[[1,2]], ycoordt.size*zcoordt.size, yz0data.shape[0], yz0data, yz0coordt, dtype=float64)
            pdfxy  = pdfxy.reshape(xcoordt.size, ycoordt.size)
            pdfyz  = pdfyz.reshape(ycoordt.size, zcoordt.size)
            # Replace nan with zero
            pdfy0 = np.nan_to_num(pdfy0)
            pdfxy, pdfyz = np.nan_to_num(pdfxy), np.nan_to_num(pdfyz)
        elif atom[0] == 1 and atom[1] == 1 and atom[2] == 0:  # consider atom-at-zero effect at both x and y
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indz0], axis=0)  # the indices of non-zero values
            coordo  = data[ind, :]
            xcoordt, ycoordt, zcoordt = xcoords[1:], ycoords[1:], zcoords[:]
            ### Compute pdfz0
            ind = np.any([indxy0, ind00], axis=0)
            z0data = zdata[ind]
            pdfz0  = estimator(1, bd[0], zcoordt.size, z0data.size, z0data, zcoordt[:,np.newaxis], dtype=float64)
            ### Compute pdfxz, pdfyz
            # Get the data for x-z and y-z panels
            ind1, ind2       = np.any([indy0,indyz0], axis=0), np.any([indx0,indxz0], axis=0)
            xz0data, yz0data = xzdata[ind1,:], yzdata[ind2,:]
            # Get all the locations to be estimated for x-z and y-z panels
            xz0xpts, xz0ypts = np.meshgrid(xcoordt, zcoordt, indexing='ij')
            xz0coordt        = np.array([xz0xpts.reshape(xz0xpts.size), xz0ypts.reshape(xz0ypts.size)]).T
            yz0xpts, yz0ypts = np.meshgrid(ycoordt, zcoordt, indexing='ij')
            yz0coordt        = np.array([yz0xpts.reshape(yz0xpts.size), yz0ypts.reshape(yz0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfxz  = estimator(2, bd[[0,2]], xcoordt.size*zcoordt.size, xz0data.shape[0], xz0data, xz0coordt, dtype=float64)
            pdfyz  = estimator(2, bd[[1,2]], ycoordt.size*zcoordt.size, yz0data.shape[0], yz0data, yz0coordt, dtype=float64)
            pdfxz  = pdfxz.reshape(xcoordt.size, zcoordt.size)
            pdfyz  = pdfyz.reshape(ycoordt.size, zcoordt.size)
            # Replace nan with zero
            pdfz0 = np.nan_to_num(pdfz0)
            pdfxz, pdfyz = np.nan_to_num(pdfxz), np.nan_to_num(pdfyz)
        elif atom[0] == 1 and atom[1] == 0 and atom[2] == 0:  # consider atom-at-zero effect at x
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indz0, indy0, indyz0], axis=0)  # the indices of non-zero values
            coordo  = data[ind, :]
            xcoordt, ycoordt, zcoordt = xcoords[1:], ycoords[:], zcoords[:]
            ### Compute pdfyz
            # Get the data for the y-z panel
            ind     = np.any([indx0,indxy0,indxz0,ind00], axis=0)
            yz0data = yzdata[ind,:]
            # Get all the locations to be estimated for y-z panels
            yz0xpts, yz0ypts = np.meshgrid(ycoordt, zcoordt, indexing='ij')
            yz0coordt        = np.array([yz0xpts.reshape(yz0xpts.size), yz0ypts.reshape(yz0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfyz  = estimator(2, bd[[1,2]], ycoords.size*zcoords.size, yz0data.shape[0], yz0data, yz0coordt, dtype=float64)
            pdfyz  = pdfyz.reshape(ycoords.size, zcoords.size)
            # Replace nan with zero
            pdfyz = np.nan_to_num(pdfyz)
        elif atom[0] == 0 and atom[1] == 1 and atom[2] == 0:  # consider atom-at-zero effect at y
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indz0, indx0, indxz0], axis=0)  # the indices of non-zero values
            coordo  = data[ind, :]
            xcoordt, ycoordt, zcoordt = xcoords[:], ycoords[1:], zcoords[:]
            ### Compute pdfyz
            # Get the data for the x-z panel
            ind     = np.any([indy0,indxy0,indyz0,ind00], axis=0)
            xz0data = xzdata[ind,:]
            # Get all the locations to be estimated for x-z panels
            xz0xpts, xz0ypts = np.meshgrid(xcoordt, zcoordt, indexing='ij')
            xz0coordt        = np.array([xz0xpts.reshape(xz0xpts.size), xz0ypts.reshape(xz0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfxz  = estimator(2, bd[[0,2]], xcoordt.size*zcoordt.size, xz0data.shape[0], xz0data, xz0coordt, dtype=float64)
            pdfxz  = pdfxz.reshape(xcoordt.size, zcoordt.size)
            # Replace nan with zero
            pdfxz = np.nan_to_num(pdfxz)
        elif atom[0] == 0 and atom[1] == 0 and atom[2] == 1:  # consider atom-at-zero effect at z
            ### Compute PDF center
            # Get the nonzero data and their coordinates range
            ind     = np.any([indn0, indy0, indx0, indxy0], axis=0)  # the indices of non-zero values
            coordo  = data[ind, :]
            xcoordt, ycoordt, zcoordt = xcoords[:], ycoords[:], zcoords[1:]
            ### Compute pdfyz
            # Get the data for the x-y panel
            ind     = np.any([indz0,indxz0,indyz0,ind00], axis=0)
            xy0data = xydata[ind,:]
            # Get all the locations to be estimated for x-z panels
            xy0xpts, xy0ypts = np.meshgrid(xcoordt, ycoordt, indexing='ij')
            xy0coordt        = np.array([xy0xpts.reshape(xy0xpts.size), xy0ypts.reshape(xy0ypts.size)]).T
            # Compute the PDF and reshape them
            pdfxy  = estimator(2, bd[[0,1]], xcoordt.size*ycoordt.size, xy0data.shape[0], xy0data, xy0coordt, dtype=float64)
            pdfxy  = pdfxy.reshape(xcoordt.size, ycoordt.size)
            # Replace nan with zero
            pdfxy = np.nan_to_num(pdfxy)

        # Compute pdfcenter
        xpts, ypts, zpts = np.meshgrid(xcoordt, ycoordt, zcoordt, indexing='ij')
        coordt     = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size), zpts.reshape(zpts.size)]).T
        No, Nt     = coordo.shape[0], coordt.shape[0]
        pdfcenter  = estimator(ndim, bd, Nt, No, coordo, coordt, dtype=float64)
        pdfcenter  = pdfcenter.reshape(xcoordt.size, ycoordt.size, zcoordt.size)
        # Replace nan with zero
        pdfcenter = np.nan_to_num(pdfcenter)

        # Combine pdfcenter, pdfxo and pdfyo to pdf
        # and compute the origin if it is a atom-at-zero effect
        if atom[0] == 0 and atom[1] == 0 and atom[2] == 0:    # no atom-at-zero effect (kx=ky=kz=1)
            pdf = pdfcenter
        elif atom[0] == 1 and atom[1] == 1 and atom[2] == 1:  # consider atom-at-zero effect at both x, y and z
            # Assign pdfcenter and the origin
            if pdfx0.sum() == 0 and pdfy0.sum() == 0 and pdfz0.sum() == 0:  # if there is no data points at the three axes then the origin pdf is set to zero
                pdf[0,0,0] = 0.
            else:
                pdf[0,0,0] = (1-kx)*(1-ky)*(1-kz)
            pdf[1:,1:,1:] = kx*ky*kz*pdfcenter
            # Assign pdfx0, pdfy0 and pdfz0
            pdf[1:,0,0] = (1-ky)*(1-kz)*kx*pdfx0
            pdf[0,1:,0] = (1-kx)*(1-kz)*ky*pdfy0
            pdf[0,0,1:] = (1-kx)*(1-ky)*kz*pdfz0
            # Assign pdfxy, pdfyz, pdfxz
            pdf[1:,1:,0] = (1-kz)*kx*ky*pdfxy
            pdf[0,1:,1:] = (1-kx)*ky*kz*pdfyz
            pdf[1:,0,1:] = (1-ky)*kx*kz*pdfxz
            # print pdfcenter.max(),pdfcenter.min()
            # print pdfx0, pdfy0, pdfz0
            # print pdfxy.max(), pdfxy.min(), pdfyz.max(), pdfyz.min(), pdfxz.max(), pdfxz.min()
        elif atom[0] == 0 and atom[1] == 1 and atom[2] == 1:  # consider atom-at-zero effect at both y and z (kx=1)
            # Assign pdfcenter
            pdf[:,1:,1:] = ky*kz*pdfcenter
            # Assign pdfx0
            pdf[:,0,0] = (1-kz)*(1-ky)*pdfx0
            # Assign pdfxy, pdfxz
            pdf[:,1:,0] = (1-kz)*ky*pdfxy
            pdf[:,0,1:] = (1-ky)*kz*pdfxz
        elif atom[0] == 1 and atom[1] == 0 and atom[2] == 1:  # consider atom-at-zero effect at both x and z (ky=1)
            # Assign pdfcenter
            pdf[1:,:,1:] = kx*kz*pdfcenter
            # Assign pdfy0
            pdf[0,:,0] = (1-kz)*(1-kx)*pdfy0
            # Assign pdfxy, pdfxz
            pdf[1:,:,0] = (1-kz)*kx*pdfxy
            pdf[0,:,1:] = (1-kx)*kz*pdfyz
        elif atom[0] == 1 and atom[1] == 1 and atom[2] == 0:  # consider atom-at-zero effect at both x and y (kz=1)
            # Assign pdfcenter
            pdf[1:,1:,:] = ky*kx*pdfcenter
            # Assign pdfz0
            pdf[0,0,:] = (1-kx)*(1-ky)*pdfz0
            # Assign pdfyz, pdfxz
            pdf[0,1:,:] = (1-kx)*ky*pdfyz
            pdf[1:,0,:] = (1-ky)*kx*pdfxz
        elif atom[0] == 1 and atom[1] == 0 and atom[2] == 0:  # consider atom-at-zero effect at x (ky=kz=1)
            # Assign pdfcenter
            pdf[1:,:,:] = kx*pdfcenter
            # Assign pdfyz
            pdf[0,:,:] = (1-kx)*pdfyz
        elif atom[0] == 0 and atom[1] == 1 and atom[2] == 0:  # consider atom-at-zero effect at y (kx=kz=1)
            # Assign pdfcenter
            pdf[:,1:,:] = ky*pdfcenter
            # Assign pdfxz
            pdf[:,0,:] = (1-ky)*pdfxz
        elif atom[0] == 0 and atom[1] == 1 and atom[2] == 1:  # consider atom-at-zero effect at z (kx=ky=1)
            # Assign pdfcenter
            pdf[:,:,1:] = kz*pdfcenter
            # Assign pdfxy
            pdf[:,:,0] = (1-kz)*pdfxy

        # Normalize pdf again in case there is slice error
        pdf = pdf/pdf.sum()

        return time.time() - t0, pdf, self.coords

    def computeBandWidth(self, data, bandwidthType):
        '''
        Compute the band width given the type.
        Input:
        bandwidthType -- the type of band width [string]
        Output: [ndarray with shape(ndim,)]
        '''
        hlist = np.zeros(self.ndim)
        for i in range(self.ndim):
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
        elif self.approach == 'kde' or self.approach == 'kde_c' or self.approach == 'kde_cuda':  # KDE
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
            kset   -- the proportion of nonzero values in each dimension [ndarray with shape (ndim,), float64]
        '''
        npts, ndim = data.shape
        thres      = .9          # The threshold to determine whether an atom-at-zero effect works

        # Compute kset
        kset = map(lambda i: np.sum(data[:, i] != 0)/float(npts), range(ndim))
        kset = np.array(kset, dtype=float64)

        # Compute atom
        # atom is set to zero if the corresponding kset is less than thres. Also,
        # If kset is larger than thres, kset is set to 1.
        atom = np.zeros(ndim, dtype=int)
        atom[kset < thres] = 1
        kset[kset > thres] = 1.

        self.kset = kset
        self.atom = atom

    def __computeEdgeCoord(self, data, nbins):
        '''
        Compute the edges and coordinates of each dimension for PDF generation.
        The first edge and coordinate are zero if the corresponding atom is True.
        Input:
            data   -- the data [numpy array with shape (npoints, ndim)]
            nbins  -- the number of bins in each dimension [list]
        Outputs:
            coords -- the coordinates for PDF generation [ndarray with shape (ndim,)]
            # edges  -- the edges for PDF generation [ndarray with shape (ndim,)]
        '''
        ndim, atom = self.ndim, self.atom
        limits     = self.limits
        coords     = []
        for i in range(ndim):
            nx              = nbins[i]
            xarray          = data[:, i]
            xedges, xcoords = np.zeros(nx+1), np.zeros(nx)
            # Get the limit
            if not atom[i] and limits is not None:
                xmin, xmax = limits[i][0], limits[i][1]
            else:
                xmin, xmax = xarray.min(), xarray.max()
            # print limits
            # print xarray
            # print xmin, xmax
            # Compute the coordinates
            if atom[i]:  # consider atom-at-zero: 1st bin is only for zero values, evenly space other bins
                xedges[1:] = np.linspace(xmin, xmax, nx)
                xcoords    = (xedges[1:] + xedges[:-1]) / 2.
                xcoords[0] = 0
            else:        # no atom-at-zero effect
                xedges     = np.linspace(xmin, xmax, nx+1)
                xcoords    = (xedges[1:] + xedges[:-1]) / 2.
            coords.append(xcoords)

        self.coords = coords

    # def computePDFKDE_old(self, data, nbins):
    #     '''
    #     Compute the PDF based on KDE.
    #     Input:
    #     data  -- the data [numpy array with shape (npoints, ndim)]
    #     nbins -- the number of bins in each dimension [list]
    #     Output:
    #     time   -- [float]
    #     pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
    #     coords -- [a numpy array with shape (ndim,)]
    #     '''
    #     para = self.approachPara.copy()
    #
    #     t0 = time.clock()
    #     # Get the band width value given the band width type
    #     if isinstance(para['bandwidth'], str):
    #         para['bandwidth'] = self.computeBandWidth(data, para['bandwidth'])
    #
    #     # Compute the sampled points based on the band width
    #     # bd = para['bandwidth']
    #     if self.ndim == 1:
    #         nx = nbins[0]
    #         xarray = data[:, 0]
    #         xmin, xmax = xarray.min(), xarray.max()
    #         coords = np.linspace(xmin, xmax, nx)
    #         samples = coords[:, np.newaxis]
    #     elif self.ndim == 2:
    #         nx, ny = nbins[0], nbins[1]
    #         xarray, yarray = data[:, 0], data[:, 1]
    #         xmin, xmax = xarray.min(), xarray.max()
    #         ymin, ymax = yarray.min(), yarray.max()
    #         xcoords, ycoords = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    #         xpts, ypts = np.meshgrid(xcoords, ycoords, indexing='ij')
    #         samples = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size)]).T
    #         coords = np.array([xcoords, ycoords])
    #     elif self.ndim == 3:
    #         nx, ny, nz = nbins[0], nbins[1], nbins[2]
    #         xarray, yarray, zarray = data[:, 0], data[:, 1], data[:, 2]
    #         xmin, xmax = xarray.min(), xarray.max()
    #         ymin, ymax = yarray.min(), yarray.max()
    #         zmin, zmax = zarray.min(), zarray.max()
    #         xcoords, ycoords, zcoords = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), np.linspace(zmin, zmax, nz)
    #         xpts, ypts, zpts = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    #         samples = np.array([xpts.reshape(xpts.size), ypts.reshape(ypts.size), zpts.reshape(zpts.size)]).T
    #         coords = np.array([xcoords, ycoords, zcoords])
    #
    #     # Compute the PDF
    #     kde_skl = KernelDensity(**para)
    #     kde_skl.fit(data)
    #     log_pdf = kde_skl.score_samples(samples)
    #     pdf = np.exp(log_pdf)
    #
    #     # Normalize the PDF and reorganize
    #     pdf = pdf / np.sum(pdf)
    #     if self.ndim == 2:
    #         # pdf = pdf.reshape(ycoords.size, xcoords.size)
    #         pdf = pdf.reshape(xcoords.size, ycoords.size)
    #     elif self.ndim == 3:
    #         # pdf = pdf.reshape(zcoords.size, ycoords.size, xcoords.size)
    #         pdf = pdf.reshape(xcoords.size, ycoords.size, zcoords.size)
    #
    #     return time.clock() - t0, pdf, coords
    #
    # def computePDFFixedBin_old(self, data, nbins):
    #     '''
    #     Compute the PDF based on the fixed bin method.
    #     Input:
    #     data  -- the data [numpy array with shape (npoints, ndim)]
    #     nbins -- the number of bins in each dimension [list]
    #     Output:
    #     time   -- [float]
    #     pdf    -- [an numpy array with shape (nsample_dim1, nsample_dim2, ...)]
    #     coords -- [a numpy array with shape (ndim,)]
    #     '''
    #     para = self.approachPara.copy()
    #     para['bins'] = nbins
    #
    #     # Compute the PDF
    #     t0 = time.clock()
    #     if self.ndim == 1:
    #         para['bins'] = nbins[0]
    #         pdf, edges = np.histogram(data, density=True, **para)
    #         coords = np.array([(edges[i+1]+edges[i])/2 for i in range(edges.size-1)])
    #     elif self.ndim == 2:
    #         x_array, y_array = data[:, 0], data[:, 1]
    #         pdf, xedges, yedges = np.histogram2d(x_array, y_array, normed=True, **para)
    #         coords = np.array([[(xedges[i+1]+xedges[i])/2 for i in range(xedges.size-1)],
    #                            [(yedges[i+1]+yedges[i])/2 for i in range(yedges.size-1)]])
    #     elif self.ndim == 3:
    #         pdf, edges = np.histogramdd(data, normed=True, **para)
    #         xedges, yedges, zedges = edges[0], edges[1], edges[2]
    #         coords = np.array([[(xedges[i+1]+xedges[i])/2 for i in range(xedges.size-1)],
    #                            [(yedges[i+1]+yedges[i])/2 for i in range(yedges.size-1)],
    #                            [(zedges[i+1]+zedges[i])/2 for i in range(zedges.size-1)]])
    #
    #     # Normalize the PDF
    #     pdf = pdf / np.sum(pdf)
    #
    #     return time.clock() - t0, pdf, coords
    #
    # def computeBandWidth_old(self, data, bandwidthType):
    #         '''
    #         Compute the band width given the type.
    #         Input:
    #         bandwidthType -- the type of band width [string]
    #         Output: [float]
    #         '''
    #         if bandwidthType == 'silverman':
    #             hlist = []
    #             for i in range(self.ndim):
    #                 xarray = data[:, i]
    #                 h = self.silverman(xarray)
    #                 hlist.append(h)
    #             return np.min(hlist)
    #         elif bandwidthType == 'crossvalidation':
    #             return self.crossValidation(data)
