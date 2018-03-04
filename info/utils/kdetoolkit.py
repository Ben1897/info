# A script for calling kde from c/cuda
#
# @Author: Peishi Jiang <Ben1897>
# @Date:   2017-02-28T09:54:51-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Ben1897
# @Last modified time: 2017-03-07T20:33:49-06:00

import ctypes
from ctypes import *
import os
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import gaussian_kde
from time import time


# Get the path of the kde.so and cuda_kde.so
dir_path = os.path.dirname(os.path.realpath(__file__))
kde_path = os.path.join(dir_path, 'kde.so')
cudakde_path = os.path.join(dir_path, 'cuda_kde.so')
cudakdegeneral_path = os.path.join(dir_path, 'cuda_kde_general.so')


allowed_kernels = ['epanechnikov', 'gaussian']
# Function for estimating PDF using c based kde
def kde_c(ndim, kernel, bd, Nt, No, coordo, coordt, dtype='float64', rtime=False):
    '''
    Calculating PDF by using KDE (Epanechnikov) based on C code.
    Inputs:
        ndim   -- the number of dimensions/variables [int]
        kernel -- the type of the kernel [str]
        bd     -- a list of bandwidths for each dimension/variable [list]
        Nt     -- the number of locations whose PDF will be estimated [int]
        No     -- the number of sampled locations [int]
        coordo -- the sampled locations [ndarray with shape(No, ndim)]
        coordt -- the locations to be estimated [ndarray with shape(Nt, ndim)]
    Outputs:
        pdf    -- the estimated pdf [ndarray with shape(Nt,)]
    '''
    # Check the kernel type
    if kernel.lower() == 'epanechnikov':
        ktype = 1
    elif kernel.lower() == 'gaussian':
        ktype = 2
    else:
        raise Exception('Unknown kernel type %s' % kernel)

    # Convert the float value of the bd into a list in a numpy array
    if ndim == 1 and isinstance(bd, float):
        bd = np.array([bd], dtype='float64')
    # Check dimensions
    if (No, ndim) != coordo.shape and (No,) != coordo.shape:
        raise Exception('Wrong dimension and size of coordo!')
    if (Nt, ndim) != coordt.shape and (Nt,) != coordt.shape:
        print Nt, ndim, coordt.shape
        raise Exception('Wrong dimension and size of coordt!')
    if len(bd) != ndim:
        raise Exception('The length of the bandwidht does not equal to the number of the dimensions!')

    # Initialize the c-based shared library
    # dll           = ctypes.CDLL('./kde.so')
    dll           = ctypes.CDLL(kde_path)
    func          = dll.kde
    func.argtypes = [c_int, c_int, c_int, c_int,
                     POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    func.restype  = POINTER(c_double)

    # Convert coordo, coordt to 1d array
    coordo, coordt = coordo.reshape(ndim*No), coordt.reshape(ndim*Nt)

    # Mapping data type to ctypes
    ndim_p   = c_int(ndim)
    Nt_p     = c_int(Nt)
    No_p     = c_int(No)
    ktype_p  = c_int(ktype)
    bd_p     = bd.ctypes.data_as(POINTER(c_double))
    coordo_p = coordo.ctypes.data_as(POINTER(c_double))
    coordt_p = coordt.ctypes.data_as(POINTER(c_double))

    # Calculate the pdf and compute the time
    start = time()
    pdf   = func(ndim_p, Nt_p, No_p, ktype_p, bd_p, coordo_p, coordt_p)
    end   = time()

    # Reshape pdf
    pdf = np.array(list(pdf[:Nt]), dtype=dtype)

    # Return results
    if rtime:
        return pdf, end - start
    else:
        return pdf

# Function for estimating PDF using c-cuda based the general multivaraite kde
def kde_cuda_general(ndim, kernel, bdinv, bddet, Nt, No, coordo, coordt, dtype='float64', rtime=False):
    '''
    Calculating PDF by using the general KDE based on C-CUDA code.
    Inputs:
        ndim   -- the number of dimensions/variables [int]
        kernel -- the type of the kernel [str]
        bdinv  -- the inverse bandwidth matrix [ndarray with shape (ndim, ndim)]
        bddet  -- the determinant of the bandwidth matrix [float]
        Nt     -- the number of locations whose PDF will be estimated [int]
        No     -- the number of sampled locations [int]
        coordo -- the sampled locations [ndarray with shape(No, ndim)]
        coordt -- the locations to be estimated [ndarray with shape(Nt, ndim)]
    Outputs:
        pdf    -- the estimated pdf [ndarray with shape(Nt,)]
    '''
    # Check the kernel type
    if kernel.lower() == 'epanechnikov':
        ktype = 1
    elif kernel.lower() == 'gaussian':
        ktype = 2
    elif kernel.lower() == 'quartic':
        ktype = 3
    else:
        raise Exception('Unknown kernel type %s' % kernel)
        # print "Only Gaussian kernel is available for computing the general multivariate KDE for now!"

    # Convert the float value of the bd into a list in a numpy array
    if ndim == 1 and isinstance(bdinv, float):
        bdinv = np.array([bdinv], dtype='float64')
    elif bdinv.shape != (ndim,ndim):
        raise Exception('Wrong dimension of the bandwidth matrix!')

    # Check dimensions
    if (No, ndim) != coordo.shape and (No,) != coordo.shape:
        raise Exception('Wrong dimension and size of coordo!')
    if (Nt, ndim) != coordt.shape and (Nt,) != coordt.shape:
        print Nt, ndim, coordt.shape
        raise Exception('Wrong dimension and size of coordt!')

    # Compute the unit volumn for the Epanechnikov kernel
    if ktype == 1:
        unitVolume = compute_unitvolumn(ndim)
    else:
        unitVolume = 0.

    # Initialize the cuda-based shared library
    # dll           = ctypes.CDLL('./cuda_kde.so')
    dll           = ctypes.CDLL(cudakdegeneral_path)
    func          = dll.cuda_kde_general
    func.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double,
                     POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    func.restype  = POINTER(c_double)

    # Convert coordo, coordt to 1d array
    coordo, coordt = coordo.reshape(ndim*No), coordt.reshape(ndim*Nt)

    # Convert bdinv to 1d array
    bdinv = bdinv.reshape(ndim*ndim)

    # Mapping data type to ctypes
    ndim_p   = c_int(ndim)
    Nt_p     = c_int(Nt)
    No_p     = c_int(No)
    ktype_p  = c_int(ktype)
    bddet_p  = c_double(bddet)
    unitVolume_p  = c_double(unitVolume)
    bdinv_p  = bdinv.ctypes.data_as(POINTER(c_double))
    coordo_p = coordo.ctypes.data_as(POINTER(c_double))
    coordt_p = coordt.ctypes.data_as(POINTER(c_double))

    # Calculate the pdf and compute the time
    start = time()
    pdf   = func(ndim_p, Nt_p, No_p, ktype_p, unitVolume_p,
                 bddet_p, bdinv_p, coordo_p, coordt_p)
    end   = time()

    # Shape pdf
    pdf = np.array(list(pdf[:Nt]), dtype=dtype)

    # Return results
    if rtime:
        return pdf, end - start
    else:
        return pdf

# Function for estimating PDF using c-cuda based kde
def kde_cuda(ndim, kernel, bd, Nt, No, coordo, coordt, dtype='float64', rtime=False):
    '''
    Calculating PDF by using KDE (Epanechnikov) based on C-CUDA code.
    Inputs:
        ndim   -- the number of dimensions/variables [int]
        kernel -- the type of the kernel [str]
        bd     -- a list of bandwidths for each dimension/variable [list]
        Nt     -- the number of locations whose PDF will be estimated [int]
        No     -- the number of sampled locations [int]
        coordo -- the sampled locations [ndarray with shape(No, ndim)]
        coordt -- the locations to be estimated [ndarray with shape(Nt, ndim)]
    Outputs:
        pdf    -- the estimated pdf [ndarray with shape(Nt,)]
    '''
    # Check the kernel type
    if kernel.lower() == 'epanechnikov':
        ktype = 1
    elif kernel.lower() == 'gaussian':
        ktype = 2
    else:
        raise Exception('Unknown kernel type %s' % kernel)

    # Convert the float value of the bd into a list in a numpy array
    if ndim == 1 and isinstance(bd, float):
        bd = np.array([bd], dtype='float64')
    # Check dimensions
    if (No, ndim) != coordo.shape and (No,) != coordo.shape:
        raise Exception('Wrong dimension and size of coordo!')
    if (Nt, ndim) != coordt.shape and (Nt,) != coordt.shape:
        print Nt, ndim, coordt.shape
        raise Exception('Wrong dimension and size of coordt!')
    if len(bd) != ndim:
        raise Exception('The length of the bandwidht does not equal to the number of the dimensions!')

    # Initialize the cuda-based shared library
    # dll           = ctypes.CDLL('./cuda_kde.so')
    dll           = ctypes.CDLL(cudakde_path)
    func          = dll.cuda_kde
    func.argtypes = [c_int, c_int, c_int, c_int,
                     POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    func.restype  = POINTER(c_double)

    # Convert coordo, coordt to 1d array
    coordo, coordt = coordo.reshape(ndim*No), coordt.reshape(ndim*Nt)

    # print np.sum(coordo > 0), np.sum(coordt > 0)
    # print coordo.size, coordt.size, ndim, Nt, No

    # Mapping data type to ctypes
    ndim_p   = c_int(ndim)
    Nt_p     = c_int(Nt)
    No_p     = c_int(No)
    ktype_p  = c_int(ktype)
    bd_p     = bd.ctypes.data_as(POINTER(c_double))
    coordo_p = coordo.ctypes.data_as(POINTER(c_double))
    coordt_p = coordt.ctypes.data_as(POINTER(c_double))

    # Calculate the pdf and compute the time
    start = time()
    pdf   = func(ndim_p, Nt_p, No_p, ktype_p, bd_p, coordo_p, coordt_p)
    end   = time()

    # Shape pdf
    pdf = np.array(list(pdf[:Nt]), dtype=dtype)

    # Return results
    if rtime:
        return pdf, end - start
    else:
        return pdf


# Function for estimating PDF using scikit-learn based kde
def kde_sklearn(ndim, kernel, bd, Nt, No, coordo, coordt, dtype='float64', rtime=False):
    '''
    Calculating PDF by using KDE (Epanechnikov) based on scikit-learn KernelDensity method.
    Inputs:
        ndim   -- the number of dimensions/variables [int]
        kernel -- the type of the kernel [str]
        bd     -- a list of bandwidths for each dimension/variable [list]
        Nt     -- the number of locations whose PDF will be estimated [int]
        No     -- the number of sampled locations [int]
        coordo -- the sampled locations [ndarray with shape(No, ndim)]
        coordt -- the locations to be estimated [ndarray with shape(Nt, ndim)]
    Outputs:
        pdf    -- the estimated pdf [ndarray with shape(Nt,)]
    '''
    # Check kernel types
    if kernel.lower() not in allowed_kernels:
        raise Exception('Unknown kernel type %s' % kernel)

    # Convert the float value of the bd into a list in a numpy array
    if ndim == 1 and isinstance(bd, float):
        bd = np.array([bd], dtype='float64')
    # Check dimensions
    if (No, ndim) != coordo.shape and (No,) != coordo.shape:
        raise Exception('Wrong dimension and size of coordo!')
    if (Nt, ndim) != coordt.shape and (Nt,) != coordt.shape:
        print Nt, ndim, coordt.shape
        raise Exception('Wrong dimension and size of coordt!')
    if len(bd) != ndim:
        raise Exception('The length of the bandwidht does not equal to the number of the dimensions!')

    # Reshape coordt when ndim is 1 and the shape is (Nt,) to the shape (Nt, 1)
    if ndim == 1 and coordt.shape == (Nt,):
        coordt = coordt[:, np.newaxis]

    # Calculate the pdf and compute the time
    start = time()
    kde_skl = KernelDensity(bandwidth=bd[0], kernel=kernel.lower())
    kde_skl.fit(coordo)
    log_pdf = kde_skl.score_samples(coordt)
    pdf = np.exp(log_pdf, dtype=dtype)

    end = time()

    # Return results
    if rtime:
        return pdf, end - start
    else:
        return pdf

# Function for estimating PDF using scipy.stats.gaussian_kde
def kde_scipy(ndim, kernel, bd, Nt, No, coordo, coordt, dtype='float64', rtime=False):
    '''
    Calculating PDF by using KDE based on scipy.stats.gaussian_kde method.
    Inputs:
        ndim   -- the number of dimensions/variables [int]
        kernel -- the type of the kernel [str]
        bd     -- a list of bandwidths for each dimension/variable [list]
        Nt     -- the number of locations whose PDF will be estimated [int]
        No     -- the number of sampled locations [int]
        coordo -- the sampled locations [ndarray with shape(No, ndim)]
        coordt -- the locations to be estimated [ndarray with shape(Nt, ndim)]
    Outputs:
        pdf    -- the estimated pdf [ndarray with shape(Nt,)]
    '''
    # Check kernel types
    if kernel.lower() not in allowed_kernels:
        raise Exception('Unknown kernel type %s' % kernel)

    # Convert the float value of the bd into a list in a numpy array
    if ndim == 1 and isinstance(bd, float):
        bd = np.array([bd], dtype='float64')
    # Check dimensions
    if (No, ndim) != coordo.shape and (No,) != coordo.shape:
        raise Exception('Wrong dimension and size of coordo!')
    if (Nt, ndim) != coordt.shape and (Nt,) != coordt.shape:
        print Nt, ndim, coordt.shape
        raise Exception('Wrong dimension and size of coordt!')
    if len(bd) != ndim:
        raise Exception('The length of the bandwidht does not equal to the number of the dimensions!')

    # Reshape coordt when ndim is 1 and the shape is (Nt,) to the shape (Nt, 1)
    if ndim == 1 and coordt.shape == (Nt,):
        coordt = coordt[:, np.newaxis]

    # Calculate the pdf and compute the time
    start = time()
    kde = gaussian_kde(coordo.T, bw_method='silverman')
    pdf = kde(coordt.T)
    end = time()

    # Return results
    if rtime:
        return pdf, end - start
    else:
        return pdf


def compute_unitvolumn(ndim):
    """Compute the volume of the unit sphere with the ndim dimensions."""
    if ndim == 1:
        return 1

    mode, remain = ndim / 2, ndim %2
    coeff1 = float(reduce(lambda x,y:x*y, range(1, mode+1)))
    coeff2 = float(reduce(lambda x,y:x*y, range(1, ndim+1)))
    if remain == 0:
        unitVolume = np.pi ** mode / coeff1
    elif remain == 1:
        unitVolume = 2. * coeff1 * (4.*np.pi)**mode / coeff2

    return unitVolume
