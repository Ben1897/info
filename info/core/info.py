# A class for calculating the statistical information
#
# 1D: H(X)
# 2D: H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
# 3D: H(X1), H(Y), H(X2), I(X1;Y), I(X1;X2), I(X2;Y), T(Y->X), II, I(X1,Y;X2), R, S, U1, U2
#
# @Author: Peishi Jiang <Ben1897>
# @Date:   2017-02-14T13:11:18-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Ben1897
# @Last modified time: 2017-02-22T21:30:56-06:00
#
# Ref:
# Allison's SUR paper

import numpy as np
import pandas as pd
from scipy.stats import entropy


class info(object):

    def __init__(self, pdfs, base=2):
        '''
        Input:
        pdf  -- a numpy array with ndim dimensions each of which has element nsample
                note: ndim in the order of [1, 2, 3]
                with shape (nsample1, nsample2, nsample3,...)
        base -- the logrithmatic base (the default is 2) [float/int]
        '''
        self.base = base

        # Check the number of variables is larger than 3 (i.e., ndim > 3)
        ndim = np.size(pdfs.shape)
        self.ndim = ndim

        # 1D
        if self.ndim == 1:
            self.__computeInfo1D(pdfs)

        # 2D
        if self.ndim == 2:
            self.__computeInfo2D(pdfs)

        # 3D
        if self.ndim == 3:
            self.__computeInfo3D(pdfs)

        # ND
        if self.ndim > 3:
            self.__computeInfoMD(pdfs)

        # Assemble all the information values into a Pandas series format
        self.__assemble()

    def __computeInfo1D(self, pdfs):
        '''
        Compute H(X)
        Input:
        pdfs -- a numpy array with shape (nx,)
        Output: NoneType
        '''
        self.hx = entropy(pdfs, base=self.base)

    def __computeInfo2D(self, pdfs):
        '''
        Compute H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
        Input:
        pdfs --  a numpy array with shape (nx, ny)
        Output: NoneType
        '''
        nx, ny         = pdfs.shape
        xpdfs, ypdfs   = np.sum(pdfs, axis=1), np.sum(pdfs, axis=0)  # p(x), p(y)
        # ypdfs_x        = pdfs / np.tile(xpdfs[:, np.newaxis], (1, ny))  # p(y|x)
        # xpdfs_y        = pdfs / np.tile(ypdfs[np.newaxis, :], (nx, 1))  # p(x|y)

        # Compute H(X) and H(Y)
        self.hx = entropy(xpdfs, base=self.base)  # H(X)
        self.hy = entropy(ypdfs, base=self.base)  # H(Y)

        # Compute H(X|Y), H(Y|X)
        self.hy_x = computeConditionalInfo(xpdfs, ypdfs, pdfs, base=2)  # H(Y|X)
        self.hx_y = computeConditionalInfo(ypdfs, xpdfs, pdfs.T, base=2)  # H(X|Y)
        # self.hxy = computeConditionalInfo_old(ypdfs, xpdfs_y.T, base=self.base)  # H(X|Y)
        # self.hyx = computeConditionalInfo_old(xpdfs, ypdfs_x, base=self.base)  # H(Y|X)

        # Compute I(X;Y)
        self.ixy = computeMutualInfo(xpdfs, ypdfs, pdfs, base=self.base)  # I(X;Y)

    def __computeInfo3D(self, pdfs):
        '''
        Compute H(X), H(Y), H(Z), I(Y;Z), I(X;Z), I(X;Y), I(Y,Z|X), I(X,Z|Y), II,
                I(X,Y;Z), R, S, U1, U2
        Here, X --> X2, Z --> Xtar, Y --> X1 in Allison's TIPNets manuscript.
        Input:
        pdfs --  a numpy array with shape (nx, ny, nz)
        Output: NoneType
        '''
        nx, ny, nz = pdfs.shape
        xpdfs, ypdfs, zpdfs    = np.sum(pdfs, axis=(1,2)), np.sum(pdfs, axis=(0,2)), np.sum(pdfs, axis=(0,1))  # p(x), p(y), p(z)
        xypdfs, yzpdfs, xzpdfs = np.sum(pdfs, axis=(2)), np.sum(pdfs, axis=(0)), np.sum(pdfs, axis=(1))  # p(x,y), p(y,z), p(x,z)

        # Compute H(X), H(Y) and H(Z)
        self.hx = entropy(xpdfs, base=self.base)  # H(X)
        self.hy = entropy(ypdfs, base=self.base)  # H(Y)
        self.hz = entropy(zpdfs, base=self.base)  # H(Z)

        # Compute I(X;Z), I(Y;Z) and I(X;Y)
        self.ixz = computeMutualInfo(xpdfs, zpdfs, xzpdfs, base=self.base)  # I(X;Z)
        self.iyz = computeMutualInfo(ypdfs, zpdfs, yzpdfs, base=self.base)  # I(Y;Z)
        self.ixy = computeMutualInfo(xpdfs, ypdfs, xypdfs, base=self.base)  # I(X;Y)

        # Compute T (transfer entropy)
        self.iyz_x = computeConditionalMutualInformation(pdfs, option=1, base=2.)  # I(Y,Z|X)
        self.ixz_y = computeConditionalMutualInformation(pdfs, option=2, base=2.)  # I(X,Z|Y)
        # self.tyz = computeTransferEntropy(xpdfs, xzpdfs, xypdfs, pdfs, base=self.base)  # T(Y->Z|X)
        # self.txz = computeTransferEntropy(ypdfs, xypdfs, yzpdfs, pdfs, base=self.base)  # T(X->Z|Y).hx_w  = self.hxw - self.hw                # H(X|W)
        # self.tyz = computeTransferEntropy_old(zpdfs_x, zpdfs_xy, pdfs, base=self.base)

        # Compute II (= I(X;Y;Z))
        self.ii = self.iyz_x - self.iyz
        self.itot = self.ii + self.ixz + self.iyz

        # Compute R(Z;X,Y)
        self.rmmi    = np.min([self.ixz, self.iyz])               # RMMI (Eq.(7) in Allison)
        self.isource = self.ixy / np.min([self.hx, self.hy])      # Is (Eq.(9) in Allison)
        self.rmin    = -self.ii if self.ii < 0 else 0             # Rmin (Eq.(10) in Allison)
        self.r       = self.rmin + self.isource*(self.rmmi-self.rmin)  # Rs (Eq.(11) in Allison)
        # self.r       = self.rmmi

        # Compute S(Z;X,Y), U(Z;X) and U(Z;Y)
        self.s = self.r + self.ii     # S (II = S - R)
        self.uxz = self.ixz - self.r  # U(X;Z) (Eq.(4) in Allison)
        self.uyz = self.iyz - self.r  # U(Y;Z) (Eq.(5) in Allison)

    def __computeInfoMD(self, pdfs):
        '''
        The function is aimed to compute the momentary interaction information at two paths and
        its corresponding conditional partial inforamtion decomposition.
        Compute I(X;Y|Z,W), I(X;Y|W), H(X|W), H(Y|W), I(X;Z|W), I(Y;Z|W)
                II(X;Z;Y|W) = I(X;Y|Z,W) - I(X;Y|W)
                Isc = I(X;Y|W) / min[H(X|W), H(Y|W)]
                RMMIc = min[I(X;Z|W), I(Y;Z|W)]
                Rminc = 0 if II > 0 else -II
                Rc = Rminc + Isc*(RMMIc - Rminc)
                Sc = II + Rc
                Uxc = I(X;Z|W) - Rc
                Uyc = I(Y:Z|W) - Rc
        Input:
        pdfs --  a numpy array with shape (nx, ny, nz, nw1, nw2, nw3,...)
        Output: NoneType
        '''
        # Compute the pdfs
        shapes = pdfs.shape
        ndims  = len(shapes)
        nx, ny, nz, nws = shapes[0], shapes[1], shapes[2], shapes[3:]
        wpdfs = np.sum(pdfs, axis=(0,1,2))   # p(w)
        xwpdfs, ywpdfs, zwpdfs = np.sum(pdfs, axis=(1,2)), np.sum(pdfs, axis=(0,2)), np.sum(pdfs, axis=(0,1))  # p(x,w), p(y,w), p(z,w)
        xywpdfs, yzwpdfs, xzwpdfs = np.sum(pdfs, axis=(2)), np.sum(pdfs, axis=(0)), np.sum(pdfs, axis=(1))  # p(x,y,w), p(y,z,w), p(x,z,w)

        # ## To be deleted
        # xpdfs, ypdfs, zpdfs = np.sum(pdfs, axis=(1,2,3)), np.sum(pdfs, axis=(0,2,3)), np.sum(pdfs, axis=(0,1,3))  # p(x), p(y), p(z)
        # xypdfs, yzpdfs, xzpdfs = np.sum(pdfs, axis=(2,3)), np.sum(pdfs, axis=(0,3)), np.sum(pdfs, axis=(1,3))  # p(x), p(y), p(z)
        # self.hx    = entropy(xpdfs.flatten(), base=self.base)    # H(W)
        # self.hy    = entropy(ypdfs.flatten(), base=self.base)    # H(W)
        # self.hz    = entropy(zpdfs.flatten(), base=self.base)    # H(W)
        # self.hxy   = entropy(xypdfs.flatten(), base=self.base)    # H(W)
        # self.hyz   = entropy(yzpdfs.flatten(), base=self.base)    # H(W)
        # self.hxz   = entropy(xzpdfs.flatten(), base=self.base)    # H(W)
        # self.ixy   = self.hx + self.hy - self.hxy
        # self.iyz   = self.hz + self.hy - self.hyz
        # self.ixz   = self.hx + self.hz - self.hxz

        # Compute all the entropies
        self.hw    = entropy(wpdfs.flatten(), base=self.base)    # H(W)
        self.hxw   = entropy(xwpdfs.flatten(), base=self.base)   # H(X,W)
        self.hyw   = entropy(ywpdfs.flatten(), base=self.base)   # H(Y,W)
        self.hzw   = entropy(zwpdfs.flatten(), base=self.base)   # H(Z,W)
        self.hxyw  = entropy(xywpdfs.flatten(), base=self.base)  # H(X,Y,W)
        self.hyzw  = entropy(yzwpdfs.flatten(), base=self.base)  # H(Y,Z,W)
        self.hxzw  = entropy(xzwpdfs.flatten(), base=self.base)  # H(X,Z,W)
        self.hxyzw = entropy(pdfs.flatten(), base=self.base)     # H(X,Y,Z,W)
        self.hx_w  = self.hxw - self.hw                # H(X|W)
        self.hy_w  = self.hyw - self.hw                # H(Y|W)

        # Compute all the conditional mutual information
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw  # I(X;Y|W)
        self.ixz_w = self.hxw + self.hzw - self.hw - self.hxzw  # I(X;Z|W)
        self.iyz_w = self.hyw + self.hzw - self.hw - self.hyzw  # I(Y;Z|W)

        # Compute MIIT
        self.ii = self.hxyw + self.hyzw + self.hxzw + self.hw - self.hxw - self.hyw - self.hzw - self.hxyzw
        self.itot = self.ii + self.ixz_w + self.iyz_w

        # Compute R(Z;X,Y|W)
        self.rmmi    = np.min([self.ixz_w, self.iyz_w])                # RMMIc
        self.isource = self.ixy_w / np.min([self.hxw, self.hyw])       # Isc
        self.rmin    = -self.ii if self.ii < 0 else 0                  # Rminc
        self.r       = self.rmin + self.isource*(self.rmmi-self.rmin)  # Rc

        # Compute S(Z;X,Y|W), U(Z;X|W) and U(Z;Y|W)
        self.s = self.r + self.ii       # Sc
        self.uxz = self.ixz_w - self.r  # U(X;Z|W)
        self.uyz = self.iyz_w - self.r  # U(Y;Z|W)

    def __assemble(self):
        '''
        Assemble all the information values into a Pandas series format
        Output: NoneType
        '''
        if self.ndim == 1:
            self.allInfo = pd.Series(self.hx, index=['H(X)'])
        elif self.ndim == 2:
            self.allInfo = pd.Series([self.hx, self.hy, self.hx_y, self.hy_x, self.ixy],
                                     index=['H(X)', 'H(Y)', 'H(X|Y)', 'H(Y|X)', 'I(X;Y)'])
        elif self.ndim == 3:
             self.allInfo = pd.Series([self.ii, self.itot, self.r, self.s, self.uxz, self.uyz,
                                       self.rmin, self.isource, self.rmmi],
                                     index=['II', 'Itotal', 'R(Z;Y,X)', 'S(Z;Y,X)', 'U(Z,X)', 'U(Z,Y)',
                                            'Rmin', 'Isource', 'RMMI'])
            # self.allInfo = pd.Series([self.hx, self.hy, self.ixz, self.iyz, self.ixy,
            #                          self.iyz_x, self.ixz_y, self.ii, self.itot, self.rmin, self.isource, self.rmmi,
            #                          self.r, self.s, self.uxz, self.uyz],
            #                          index=['H(X)', 'H(Y)', 'I(X;Z)', 'I(Y;Z)', 'I(X;Y)',
            #                                 'I(Y,Z|X)', 'I(X,Z|Y)', 'II', 'Itotal', 'Rmin', 'Isource', 'RMMI',
            #                                 'R(Z;Y,X)', 'S(Z;Y,X)', 'U(Z,X)', 'U(Z,Y)'])
        else:
            self.allInfo = pd.Series([self.ii, self.itot, self.r, self.s, self.uxz, self.uyz,
                                      self.rmmi, self.isource, self.rmin],
                                     index=['MIIT', 'Itotal', 'Rc', 'Sc', 'Uxc', 'Uyc',
                                            'RMMIc', 'Isc', 'Rminc'])


def computeConditionalInfo(xpdfs, ypdfs, xypdfs, base=2):
    '''
    Compute the conditional information H(Y|X)
    Input:
    xpdfs  -- pdf of x [a numpy array with shape(nx)]
    ypdfs  -- pdf of y [a numpy array with shape(ny)]
    xypdfs -- joint pdf of y and x [a numpy array with shape (nx, ny)]
    Output:
    the coonditional information [float]
    '''
    nx, ny = xypdfs.shape

    xpdfs1d = np.copy(xpdfs)

    # Expand xpdfs and ypdfs into shape (nx, ny)
    xpdfs = np.tile(xpdfs[:, np.newaxis], [1, ny])
    ypdfs = np.tile(ypdfs[np.newaxis, :], [nx, 1])

    # Calculate the log of p(x,y)/p(x) and treat log(0) as zero
    ypdfs_x_log, ypdfs_x = np.ma.log(xypdfs/xpdfs), np.ma.divide(xypdfs, xpdfs)
    ypdfs_x_log, ypdfs_x = ypdfs_x_log.filled(0), ypdfs_x.filled(0)

    # Get the each info element in H(Y|X=x)
    hy_x_xy = - ypdfs_x * ypdfs_x_log / np.log(base)

    # Sum hxy_xy over y to get H(Y|X=x)
    hy_x_x = np.sum(hy_x_xy, axis=1)

    # Calculate H(Y|X)
    return np.sum(xpdfs1d*hy_x_x)


def computeMutualInfo(xpdfs, ypdfs, pdfs, base=2):
    '''
    Compute the mutual information I(X;Y)
    Input:
    xpdfs  -- pdf of x [a numpy array with shape (nx)]
    ypdfs  -- pdf of y [a numpy array with shape (ny)]
    pdfs -- the joint pdf of x and y [a numpy array with shape (nx, ny)]
    Output:
    the mutual information [float]
    '''
    nx, ny = pdfs.shape

    # Expand xpdfs and ypdfs to the shape (nx, ny)
    xpdfs = np.tile(xpdfs[:, np.newaxis], (1, ny))
    ypdfs = np.tile(ypdfs[np.newaxis, :], (nx, 1))

    # Calculate log(p(x,y)/(p(x)*p(y)))
    ixypdf_log = np.ma.log(pdfs/(xpdfs*ypdfs))
    ixypdf_log = ixypdf_log.filled(0)

    # Calculate each info element in I(X;Y)
    ixy_xy = pdfs * ixypdf_log / np.log(base)

    # Calculate mutual information
    return np.sum(ixy_xy)


def computeConditionalMutualInformation(pdfs, option=1, base=2.):
    '''
    Compute the transfer entropy T(Y->Z|X) or conditional mutual information I(Y,Z|X)
    Input:
    pdfs   -- the joint pdf of x, y and z [a numpy array with shape (nx, ny, nz)]
    option -- 1: I(Y,Z|X); 2: I(X,Z|Y)
    base   -- the log base [float]
    Output:
    the transfer entropy [float]
    '''
    nx, ny, nz = pdfs.shape
    xpdfs, ypdfs, zpdfs    = np.sum(pdfs, axis=(1,2)), np.sum(pdfs, axis=(0,2)), np.sum(pdfs, axis=(0,1))  # p(x), p(y), p(z)
    xypdfs, yzpdfs, xzpdfs = np.sum(pdfs, axis=(2)), np.sum(pdfs, axis=(0)), np.sum(pdfs, axis=(1))  # p(x,y), p(y,z), p(x,z)

    if option == 1:  # T(Y->Z|X)
        # Expand zpdfs, xzpdfs, yzpdfs to the shape (nx, ny, nz)
        factor1 = np.tile(xpdfs[:, np.newaxis, np.newaxis], [1, ny, nz])
        factor2 = np.tile(xzpdfs[:, np.newaxis, :], [1, ny, 1])
        factor3 = np.tile(xypdfs[:, :, np.newaxis], [1, 1, nz])
    elif option == 2:  # T(Y->Z|X)
        # Expand zpdfs, xzpdfs, yzpdfs to the shape (nx, ny, nz)
        factor1 = np.tile(ypdfs[np.newaxis, :, np.newaxis], [nx, 1, nz])
        factor2 = np.tile(yzpdfs[np.newaxis, :, :], [nx, 1, 1])
        factor3 = np.tile(xypdfs[:, :, np.newaxis], [1, 1, nz])

    # Calculate log(p(y|z,x)/p(y|x))
    txypdf_log = np.ma.log(pdfs*factor1/(factor2*factor3))
    txypdf_log = txypdf_log.filled(0)

    # Calculate each info element in T(Y->Z|X)
    txypdf = pdfs * txypdf_log / np.log(base)

    # Calculate the transfer entropy
    return np.sum(txypdf)


def computeConditionalInfo_old(xpdfs, ypdfs_x, base=2):
    '''
    Compute the conditional information H(Y|X)
    Input:
    xpdfs   -- pdf of x [a numpy array with shape(nx)]
    ypdfs_x -- conditional pdf of y on x [a numpy array with shape (nx, ny)]
    Output:
    the coonditional information [float]
    '''
    nx, ny = ypdfs_x.shape

    # Calculate the log of p(y|x) and treat log(0) as zero
    ypdfs_x_log = np.ma.log(ypdfs_x)
    ypdfs_x_log = ypdfs_x_log.filled(0)

    # Get the each info element in H(Y|X=x)
    hy_x_xy = - ypdfs_x * ypdfs_x_log / np.log(base)

    # Sum hxy_xy over y to get H(Y|X=x)
    hy_x_x = np.sum(hy_x_xy, axis=1)

    # Calculate H(Y|X)
    return np.sum(xpdfs*hy_x_x)


def computeTransferEntropy_old(ypdfs_x, ypdfs_xz, pdfs, base=2):
    '''
    Compute the transfer entropy T(Y->Z|X) or conditional information information I(Y,Z|X)
    Input:
    ypdfs_x  -- conditional pdf of y on x [a numpy array with shape (nx, ny)]
    ypdfs_xz -- conditional pdf of y on x and z [a numpy array with shape (nx, ny, nz)]
    pdfs     -- the joint pdf of x, y and z [a numpy array with shape (nx, ny, nz)]
    Output:
    the transfer entropy [float]
    '''
    nx, ny, nz = pdfs.shape

    # Expand ypdfs_x to the shape (nx, ny, nz)
    ypdfs_x = np.tile(ypdfs_x[:, :, np.newaxis], [1, 1, nz])

    # Calculate log(p(y|z,x)/p(y|x))
    txypdf_log = np.ma.log(ypdfs_xz/ypdfs_x)
    txypdf_log = txypdf_log.filled(0)

    # Calculate each info element in T(Y->Z|X)
    txypdf = pdfs * txypdf_log

    # Calculate the transfer entropy
    return np.sum(txypdf)


def computeTransferEntropy_old2(xpdfs, xzpdfs, xypdfs, pdfs, base=2):
    '''
    Compute the transfer entropy T(Y->Z|X) or conditional information information I(Y,Z|X)
    Input:
    zpdfs  -- pdf of z [a numpy array with shape (nx)]
    xzpdfs -- joint pdf of x and z [a numpy array with shape (nx, nz)]
    yzpdfs -- joint pdf of y and z [a numpy array with shape (ny, nz)]
    pdfs   -- the joint pdf of x, y and z [a numpy array with shape (nx, ny, nz)]
    Output:
    the transfer entropy [float]
    '''
    nx, ny, nz = pdfs.shape

    # Expand zpdfs, xzpdfs, yzpdfs to the shape (nx, ny, nz)
    xpdfs  = np.tile(xpdfs[:, np.newaxis, np.newaxis], [1, ny, nz])
    xzpdfs = np.tile(xzpdfs[:, np.newaxis, :], [1, ny, 1])
    xypdfs = np.tile(xypdfs[:, :, np.newaxis], [1, 1, nz])

    # Calculate log(p(y|z,x)/p(y|x))
    txypdf_log = np.ma.log(pdfs*xpdfs/(xzpdfs*xypdfs))
    txypdf_log = txypdf_log.filled(0)

    # Calculate each info element in T(Y->Z|X)
    txypdf = pdfs * txypdf_log / np.log(base)

    # Calculate the transfer entropy
    return np.sum(txypdf)
