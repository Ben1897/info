# A class for calculating the statistical information
#
# 1D: H(X)
# 2D: H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
# 3D: H(X1), H(Y), H(X2), I(X1;Y), I(X1;X2), I(X2;Y), T(Y->X), II, I(X1,Y;X2), R, S, U1, U2
#
#
# Ref:
# Allison's SUR paper

import numpy as np
import pandas as pd
from scipy.stats import entropy


class info(object):

    def __init__(self, ndim, pdfs, base=2, conditioned=False, specific=False, MPID2=False):
        '''
        Input:
        ndim -- the number of dimension to be computed [int]
        pdfs  -- if MPID2 is False, a numpy array with ndim dimensions each of which has element nsample
                          note: ndim in the order of [1, 2, 3]
                          with shape (nsample1, nsample2, nsample3,...)
                 if MPID2 is True, a list of three pdfs,
                          each of which has the same format as the one when MPID2 is False
        base -- the logrithmatic base (the default is 2) [float/int]
        conditioned -- whether including conditions [bool]
        specific -- whether calculating the specific PID [bool]
        MPID2 -- whether calculating MPID2 [bool]
                 if False, compute info by using computeInfo1D, computeInfo2D, computeInfo3D, computeInfoMD,
                 if True, compute info by using computeInfoMD2
        '''
        self.base = base
        self.conditioned = conditioned
        self.specific = specific

        if MPID2:
            try:
                pdfs, pdfs1, pdfs2 = pdfs[0], pdfs[1], pdfs[2]
            except:
                raise Exception("Cannot parse pdfs properly when MPID2 is True.")
            self.__computeInfoMD2(pdfs, pdfs1, pdfs2)
            return

        self.ndim = ndim

        # 1D
        if self.ndim == 1 and not conditioned:
            self.__computeInfo1D(pdfs)
        elif self.ndim == 1 and conditioned:
            self.__computeInfo1D_conditioned(pdfs)

        # 2D
        if self.ndim == 2 and not conditioned:
            self.__computeInfo2D(pdfs)
        elif self.ndim == 2 and conditioned:
            self.__computeInfo2D_conditioned(pdfs)

        # 3D
        if self.ndim == 3 and not conditioned and not specific:
            self.__computeInfo3D(pdfs)
        elif self.ndim == 3 and not conditioned and specific:
            self.__computeInfo3D_specific(pdfs)
        elif self.ndim == 3 and conditioned:
            self.__computeInfo3D_conditioned(pdfs)

        # Assemble all the information values into a Pandas series format
        self.__assemble()

    def __computeInfo1D(self, pdfs):
        '''
        Compute H(X)
        Input:
        pdfs -- a numpy array with shape (nx,)
        Output: NoneType
        '''
        self.hx = computeEntropy(pdfs, base=self.base)

    def __computeInfo1D_conditioned(self, pdfs):
        '''
        Compute H(X|W)
        '''
        # Compute the pdfs
        shapes = pdfs.shape
        ndims  = len(shapes)
        nx, nws = shapes[0], shapes[1:]
        wpdfs = np.sum(pdfs, axis=(0))   # p(w)
        xpdfs = np.sum(pdfs, axis=tuple(range(1,ndims)))

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs.flatten(), base=self.base)    # H(W)
        self.hx    = computeEntropy(xpdfs.flatten(), base=self.base)    # H(X)
        self.hxw   = computeEntropy(pdfs.flatten(), base=self.base)   # H(X,W)
        self.hx_w  = self.hxw - self.hw                # H(X|W)

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
        self.hx = computeEntropy(xpdfs, base=self.base)  # H(X)
        self.hy = computeEntropy(ypdfs, base=self.base)  # H(Y)

        # Compute H(X|Y), H(Y|X)
        self.hy_x = computeConditionalInfo(xpdfs, ypdfs, pdfs, base=2)  # H(Y|X)
        self.hx_y = computeConditionalInfo(ypdfs, xpdfs, pdfs.T, base=2)  # H(X|Y)
        # self.hxy = computeConditionalInfo_old(ypdfs, xpdfs_y.T, base=self.base)  # H(X|Y)
        # self.hyx = computeConditionalInfo_old(xpdfs, ypdfs_x, base=self.base)  # H(Y|X)

        # Compute I(X;Y)
        self.ixy = computeMutualInfo(xpdfs, ypdfs, pdfs, base=self.base)  # I(X;Y)

        # self.hxy = computeEntropy(pdfs, base=self.base)
        # print self.hy_x - (self.hxy - self.hx)
        # print self.hx_y - (self.hxy - self.hy)
        # print self.ixy - (self.hy + self.hx - self.hxy)

    def __computeInfo2D_conditioned(self, pdfs):
        '''
        Compute H(X|W), H(Y|W), H(X,Y|W), I(X,Y|W)
        '''
        # Compute the pdfs
        shapes = pdfs.shape
        ndims  = len(shapes)
        nx, ny, nws = shapes[0], shapes[1], shapes[2:]
        wpdfs = np.sum(pdfs, axis=(0,1))   # p(w)
        xwpdfs, ywpdfs = np.sum(pdfs, axis=(1)), np.sum(pdfs, axis=(0)) # p(x,w), p(y,w)
        xpdfs, ypdfs = np.sum(pdfs, axis=tuple(range(1,ndims))), np.sum(pdfs, axis=tuple([0]+range(2,ndims)))
        xypdfs = np.sum(pdfs, axis=tuple(range(2,ndims)))

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs.flatten(), base=self.base)    # H(W)
        self.hx    = computeEntropy(xpdfs.flatten(), base=self.base)    # H(X)
        self.hy    = computeEntropy(ypdfs.flatten(), base=self.base)    # H(Y)
        self.hxy   = computeEntropy(xypdfs.flatten(), base=self.base)   # H(X,Y)
        self.hxw   = computeEntropy(xwpdfs.flatten(), base=self.base)   # H(X,W)
        self.hyw   = computeEntropy(ywpdfs.flatten(), base=self.base)   # H(Y,W)
        self.hxyw  = computeEntropy(pdfs.flatten(), base=self.base)  # H(X,Y,W)
        self.hx_w  = self.hxw - self.hw                # H(X|W)
        self.hy_w  = self.hyw - self.hw                # H(Y|W)
        self.hx_y  = self.hxy - self.hy
        self.hy_x  = self.hxy - self.hx

        # Compute all the conditional mutual information
        self.ixy = self.hx - self.hx_y
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw  # I(X;Y|W)

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
        self.hx = computeEntropy(xpdfs, base=self.base)  # H(X)
        self.hy = computeEntropy(ypdfs, base=self.base)  # H(Y)
        self.hz = computeEntropy(zpdfs, base=self.base)  # H(Z)
        self.hxy = computeEntropy(xypdfs.flatten(), base=self.base)  # H(X,Y)
        self.hyz = computeEntropy(yzpdfs.flatten(), base=self.base)  # H(Y,Z)
        self.hxz = computeEntropy(xzpdfs.flatten(), base=self.base)  # H(X,Z)
        self.hxyz = computeEntropy(pdfs.flatten(), base=self.base)  # H(X,Z)

        # Compute I(X;Z), I(Y;Z) and I(X;Y)
        # self.ixz = computeMutualInfo(xpdfs, zpdfs, xzpdfs, base=self.base)  # I(X;Z)
        # self.iyz = computeMutualInfo(ypdfs, zpdfs, yzpdfs, base=self.base)  # I(Y;Z)
        # self.ixy = computeMutualInfo(xpdfs, ypdfs, xypdfs, base=self.base)  # I(X;Y)
        self.ixy = self.hx + self.hy - self.hxy
        self.ixz = self.hx + self.hz - self.hxz
        self.iyz = self.hy + self.hz - self.hyz

        # Compute T (transfer entropy)
        # self.iyz_x = computeConditionalMutualInformation(pdfs, option=1, base=2.)  # I(Y,Z|X)
        # self.ixz_y = computeConditionalMutualInformation(pdfs, option=2, base=2.)  # I(X,Z|Y)
        # self.tyz = computeTransferEntropy(xpdfs, xzpdfs, xypdfs, pdfs, base=self.base)  # T(Y->Z|X)
        # self.txz = computeTransferEntropy(ypdfs, xypdfs, yzpdfs, pdfs, base=self.base)  # T(X->Z|Y).hx_w  = self.hxw - self.hw                # H(X|W)
        # self.tyz = computeTransferEntropy_old(zpdfs_x, zpdfs_xy, pdfs, base=self.base)

        # Compute II (= I(X;Y;Z))
        # self.ii = self.iyz_x - self.iyz
        self.itot = self.hxy + self.hz - self.hxyz # I(X,Y;Z)
        self.ii = self.itot - self.ixz - self.iyz  # interaction information

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


    def __computeInfo3D_specific(self, pdfs):
        '''
        The function is aimed to compute the specific partial information decomposition.
        Compute s(X=x,Y=y), r(X=x, Y=y), ux(X=x, Y=y), uy(X=x, Y=y)
        Input:
        pdfs --  a numpy array with shape (nx, ny, nz, nw1, nw2, nw3,...)
        Output: NoneType
        '''
        base = self.base
        nx, ny, nz = pdfs.shape
        xpdfs, ypdfs, zpdfs    = np.sum(pdfs, axis=(1,2)), np.sum(pdfs, axis=(0,2)), np.sum(pdfs, axis=(0,1))  # p(x), p(y), p(z)
        xypdfs, yzpdfs, xzpdfs = np.sum(pdfs, axis=(2)), np.sum(pdfs, axis=(0)), np.sum(pdfs, axis=(1))  # p(x,y), p(y,z), p(x,z)

        # Compute the conditional probability and mask any nan values
        xpdfs_ex = np.tile(xpdfs[:, np.newaxis], [1, nz])
        ypdfs_ex = np.tile(ypdfs[:, np.newaxis], [1, nz])
        xypdfs_ex = np.tile(xypdfs[:, :, np.newaxis], [1, 1, nz])
        z_xpdfs, z_ypdfs = np.ma.divide(xzpdfs, xpdfs_ex).filled(0), np.ma.divide(yzpdfs, ypdfs_ex).filled(0)  # p(Z|X), p(Z|Y)
        z_xypdfs = np.ma.divide(pdfs, xypdfs_ex).filled(0) # p(Z|X, Y)
        xpdfs_ex2 = np.tile(xpdfs[np.newaxis, :], [nx, 1])
        ypdfs_ex2 = np.tile(ypdfs[np.newaxis, :], [nx, 1])
        x_ypdfs, y_xpdfs = np.ma.divide(xypdfs, xpdfs_ex2).filled(0), np.ma.divide(xypdfs, ypdfs_ex2).filled(0)  # p(X|Y), p(Y|X)

        # Extend some pdfs to 3D
        z_xpdfs, z_ypdfs = np.tile(z_xpdfs[:, np.newaxis, :], [1, ny, 1]), np.tile(z_ypdfs[np.newaxis, :, :], [nx, 1, 1])
        xypdfs, yzpdfs, xzpdfs = np.tile(xypdfs[:, :, np.newaxis], [1, 1, nz]), np.tile(yzpdfs[np.newaxis, :, :], [nx, 1, 1]), \
                                 np.tile(xzpdfs[:, np.newaxis, :], [1, ny, 1])
        x_ypdfs, y_xpdfs = np.tile(x_ypdfs[:, :, np.newaxis], [1, 1, nz]), np.tile(y_xpdfs[:, :, np.newaxis], [1, 1, nz])
        xpdfs, ypdfs, zpdfs = np.tile(xpdfs[:, np.newaxis, np.newaxis], [1, ny, nz]), np.tile(ypdfs[np.newaxis, :, np.newaxis], [nx, 1, nz]), \
                              np.tile(zpdfs[np.newaxis, np.newaxis, :], [nx, ny, 1])

        # Initialize the set for s, r, ux and uy
        self.ss, self.rs = np.zeros([nx, ny]), np.zeros([nx, ny])
        self.uxzs, self.uyzs = np.zeros([nx, ny]), np.zeros([nx, ny])
        self.rmins, self.rmmis = np.zeros([nx, ny]), np.zeros([nx, ny])
        self.isources = np.zeros([nx, ny])

        # Initialize the specific mutual information and the specific interaction information
        self.ixsz = np.zeros([nx, ny])  # I(X->Z)
        self.iysz = np.zeros([nx, ny])  # I(Y->Z)
        self.itots = np.zeros([nx, ny])  # I(X=x,Y=y;Z)
        self.iis = np.zeros([nx, ny])  # II(X=x;Y=y;Z)

        # Initialize some utility matrices
        indicator1 = np.zeros([nx, ny])
        indicator2 = np.zeros([nx, ny])
        sign = np.zeros([nx, ny])

        # Compute the specific contextual information
        # Compute I(X->Z)
        plog1 = np.ma.log(z_xpdfs).filled(0) / np.log(base)
        term1 = np.sum(z_xypdfs*plog1, axis=2)
        plog2 = np.ma.log(zpdfs).filled(0) / np.log(base)
        term2 = np.sum(z_xypdfs*plog2, axis=2)
        self.ixsz = term1 - term2

        # Compute I(Y->Z)
        plog1 = np.ma.log(z_ypdfs).filled(0) / np.log(base)
        term1 = np.sum(z_xypdfs*plog1, axis=2)
        plog2 = np.ma.log(zpdfs).filled(0) / np.log(base)
        term2 = np.sum(z_xypdfs*plog2, axis=2)
        self.iysz = term1 - term2

        # Compute I(X=x, Y=y; Z) and II(X=x; Y=y; Z)
        plog1 = np.ma.log(z_xypdfs).filled(0) / np.log(base)
        term1 = np.sum(z_xypdfs*plog1, axis=2)
        plog2 = np.ma.log(zpdfs).filled(0) / np.log(base)
        term2 = np.sum(z_xypdfs*plog2, axis=2)
        self.itots = term1 - term2
        self.iis = self.itots - self.ixsz - self.iysz

        # Compute the specific PID
        indicator1[np.where(np.logical_not(equal(xypdfs[:,:,0], xpdfs[:,:,0]*ypdfs[:,:,0])))] = 1.  # indicate whether p(x) and p(y) are independent
        self.isources = indicator1 * np.maximum(x_ypdfs[:,:,0], y_xpdfs[:,:,0])
        indicator2[np.where(self.ixsz*self.iysz > 0)] = 1.  # indicate whether I(X->Z) and I(Y->Z) are different signs
        sign = np.sign(self.ixsz)
        self.rmmis = indicator2 * sign * np.minimum(np.abs(self.ixsz), np.abs(self.iysz))  # rb
        self.rs = self.isources * self.rmmis
        self.ss = self.iis + self.rs
        self.uxzs = self.ixsz - self.rs
        self.uyzs = self.iysz - self.rs

        # Compute the expectation of SPID
        self.r = np.sum(xypdfs[:,:,0]*self.rs)
        self.s = np.sum(xypdfs[:,:,0]*self.ss)
        self.uxz = np.sum(xypdfs[:,:,0]*self.uxzs)
        self.uyz = np.sum(xypdfs[:,:,0]*self.uyzs)
        self.rmin = np.sum(xypdfs[:,:,0]*self.rmins)
        self.rmmi = np.sum(xypdfs[:,:,0]*self.rmmis)
        self.isource = np.sum(xypdfs[:,:,0]*self.isources)
        self.itot = np.sum(xypdfs[:,:,0]*self.itots)
        self.ii = np.sum(xypdfs[:,:,0]*self.iis)


    def __computeInfo3D_specific_wrong(self, pdfs):
        '''
        The function is aimed to compute the specific partial information decomposition.
        Compute s(X=x,Y=y), r(X=x, Y=y), ux(X=x, Y=y), uy(X=x, Y=y)
        Input:
        pdfs --  a numpy array with shape (nx, ny, nz, nw1, nw2, nw3,...)
        Output: NoneType
        '''
        base = self.base
        nx, ny, nz = pdfs.shape
        xpdfs, ypdfs, zpdfs    = np.sum(pdfs, axis=(1,2)), np.sum(pdfs, axis=(0,2)), np.sum(pdfs, axis=(0,1))  # p(x), p(y), p(z)
        xypdfs, yzpdfs, xzpdfs = np.sum(pdfs, axis=(2)), np.sum(pdfs, axis=(0)), np.sum(pdfs, axis=(1))  # p(x,y), p(y,z), p(x,z)

        # Compute the conditional probability and mask any nan values
        xpdfs_ex = np.tile(xpdfs[:, np.newaxis], [1, nz])
        ypdfs_ex = np.tile(ypdfs[:, np.newaxis], [1, nz])
        xypdfs_ex = np.tile(xypdfs[:, :, np.newaxis], [1, 1, nz])
        z_xpdfs, z_ypdfs = np.ma.divide(xzpdfs, xpdfs_ex).filled(0), np.ma.divide(yzpdfs, ypdfs_ex).filled(0)  # p(Z|X), p(Z|Y)
        print z_xpdfs
        # print z_ypdfs
        z_xypdfs = np.ma.divide(pdfs, xypdfs_ex).filled(0) # p(Z|X, Y)
        # print z_xypdfs
        xpdfs_ex2 = np.tile(xpdfs[np.newaxis, :], [nx, 1])
        ypdfs_ex2 = np.tile(ypdfs[np.newaxis, :], [nx, 1])
        x_ypdfs, y_xpdfs = np.ma.divide(xypdfs, xpdfs_ex2).filled(0), np.ma.divide(xypdfs, ypdfs_ex2).filled(0)  # p(X|Y), p(Y|X)

        # Initialize the set for s, r, ux and uy
        self.ss, self.rs = np.zeros([nx, ny]), np.zeros([nx, ny])
        self.uxzs, self.uyzs = np.zeros([nx, ny]), np.zeros([nx, ny])
        self.rmins, self.rmmis = np.zeros([nx, ny]), np.zeros([nx, ny])
        self.isources = np.zeros([nx, ny])

        # Initialize the specific mutual information and the specific interaction information
        self.ixsz = np.zeros(nx)  # i(X=x;Z)
        self.iysz = np.zeros(ny)  # i(Y=y;Z)
        self.itots = np.zeros([nx, ny])  # i(X=x,Y=y;Z)
        self.iis = np.zeros([nx, ny])  # ii(X=x;Y=y;Z)

        # Compute the specific information
        # Compute i(X=x; Z)
        for x in range(nx):
            z_xspdfs = z_xpdfs[x, :]  # P(Z|X=x)
            # print z_xspdfs
            # Compute the first term
            plog1 = np.ma.log(z_xspdfs).filled(0) / np.log(base)
            term1 = np.sum(z_xspdfs * plog1)
            # Compute the second term
            plog2 = np.ma.log(zpdfs).filled(0) / np.log(base)
            term2 = np.sum(z_xspdfs * plog2)
            self.ixsz[x] = term1 - term2

        # Compute i(Y=y; Z)
        for y in range(ny):
            z_yspdfs = z_ypdfs[y, :]  # P(Z|Y=y)
            # Compute the first term
            plog1 = np.ma.log(z_yspdfs).filled(0) / np.log(base)
            term1 = np.sum(z_yspdfs * plog1)
            # Compute the second term
            plog2 = np.ma.log(zpdfs).filled(0) / np.log(base)
            term2 = np.sum(z_yspdfs * plog2)
            self.iysz[y] = term1 - term2

        # Compute i(X=x, Y=y; Z) and ii(X=x; Y=y; Z)
        for x in range(nx):
            for y in range(ny):
                z_xyspdfs = z_xypdfs[x, y, :]  # P(Z|X=x, Y=y)
                # Compute the first term
                plog1 = np.ma.log(z_xyspdfs).filled(0) / np.log(base)
                term1 = np.sum(z_xyspdfs * plog1)
                # Compute the second term
                plog2 = np.ma.log(zpdfs).filled(0) / np.log(base)
                term2 = np.sum(z_xyspdfs * plog2)
                # if x+1 == 1 and y+1 == 1:
                #     print z_xyspdfs
                #     print zpdfs
                #     print z_xyspdfs * plog1 - z_xyspdfs * plog2
                # if x+1 == 3 and y+1 == 4:
                #     print z_xyspdfs
                #     print zpdfs
                #     print z_xyspdfs * plog1 - z_xyspdfs * plog2
                self.itots[x, y] = term1 - term2
                self.iis[x, y] = self.itots[x, y] - self.ixsz[x] - self.iysz[y]

        # Compute the specific PID
        for x in range(nx):
            for y in range(ny):
                # Compute rmin, rmmi
                self.rmins[x, y] = 0. if self.iis[x, y] > 0  else -self.iis[x, y]
                self.rmmis[x, y] = np.min([self.ixsz[x], self.iysz[y]])
                # Compute isource
                indicator = 0. if equal(xypdfs[x,y], xpdfs[x]*ypdfs[y]) else 1.
                self.isources[x, y] = indicator * np.max([x_ypdfs[x, y], y_xpdfs[x, y]])
                # Compute r
                self.rs[x, y] = (1-self.isources[x, y]) * self.rmins[x, y] + self.isources[x, y] * self.rmmis[x, y]
                # Compute s, uxz, uyz
                self.ss[x, y] = self.iis[x, y] + self.rs[x, y]
                self.uxzs[x, y] = self.ixsz[x] - self.rs[x, y]
                self.uyzs[x, y] = self.iysz[y] - self.rs[x, y]

        # Compute the expectation of SPID
        self.r = np.sum(xypdfs*self.rs)
        self.s = np.sum(xypdfs*self.ss)
        self.uxz = np.sum(xypdfs*self.uxzs)
        self.uyz = np.sum(xypdfs*self.uyzs)
        self.rmin = np.sum(xypdfs*self.rmins)
        self.rmmi = np.sum(xypdfs*self.rmmis)
        self.isource = np.sum(xypdfs*self.isources)
        self.itot = np.sum(xypdfs*self.itots)
        self.ii = np.sum(xypdfs*self.iis)


    def __computeInfo3D_conditioned(self, pdfs):
        '''
        The function is aimed to compute the momentary interaction information at two paths and
        its corresponding momentary inforamtion partitioning.
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
        xpdfs, ypdfs, zpdfs = np.sum(pdfs, axis=tuple(range(1,ndims))), np.sum(pdfs, axis=tuple([0]+range(2,ndims))), np.sum(pdfs, axis=tuple([0,1]+range(3,ndims)))
        # xypdfs = np.sum(pdfs, axis=tuple(range(2,ndims)))

        # ## To be deleted
        # xpdfs, ypdfs, zpdfs = np.sum(pdfs, axis=(1,2,3)), np.sum(pdfs, axis=(0,2,3)), np.sum(pdfs, axis=(0,1,3))  # p(x), p(y), p(z)
        # xypdfs, yzpdfs, xzpdfs = np.sum(pdfs, axis=(2,3)), np.sum(pdfs, axis=(0,3)), np.sum(pdfs, axis=(1,3))  # p(x), p(y), p(z)
        # self.hx    = computeEntropy(xpdfs.flatten(), base=self.base)    # H(W)
        # self.hy    = computeEntropy(ypdfs.flatten(), base=self.base)    # H(W)
        # self.hz    = computeEntropy(zpdfs.flatten(), base=self.base)    # H(W)
        # self.hxy   = computeEntropy(xypdfs.flatten(), base=self.base)    # H(W)
        # self.hyz   = computeEntropy(yzpdfs.flatten(), base=self.base)    # H(W)
        # self.hxz   = computeEntropy(xzpdfs.flatten(), base=self.base)    # H(W)
        # self.ixy   = self.hx + self.hy - self.hxy
        # self.iyz   = self.hz + self.hy - self.hyz
        # self.ixz   = self.hx + self.hz - self.hxz

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs.flatten(), base=self.base)    # H(W)
        self.hx    = computeEntropy(xpdfs.flatten(), base=self.base)    # H(X)
        self.hy    = computeEntropy(ypdfs.flatten(), base=self.base)    # H(Y)
        self.hz    = computeEntropy(zpdfs.flatten(), base=self.base)    # H(Z)
        self.hxw   = computeEntropy(xwpdfs.flatten(), base=self.base)   # H(X,W)
        self.hyw   = computeEntropy(ywpdfs.flatten(), base=self.base)   # H(Y,W)
        self.hzw   = computeEntropy(zwpdfs.flatten(), base=self.base)   # H(Z,W)
        self.hxyw  = computeEntropy(xywpdfs.flatten(), base=self.base)  # H(X,Y,W)
        self.hyzw  = computeEntropy(yzwpdfs.flatten(), base=self.base)  # H(Y,Z,W)
        self.hxzw  = computeEntropy(xzwpdfs.flatten(), base=self.base)  # H(X,Z,W)
        self.hxyzw = computeEntropy(pdfs.flatten(), base=self.base)     # H(X,Y,Z,W)
        self.hx_w  = self.hxw - self.hw                # H(X|W)
        self.hy_w  = self.hyw - self.hw                # H(Y|W)

        # Compute all the conditional mutual information
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw  # I(X;Y|W)
        self.ixz_w = self.hxw + self.hzw - self.hw - self.hxzw  # I(X;Z|W)
        self.iyz_w = self.hyw + self.hzw - self.hw - self.hyzw  # I(Y;Z|W)

        ## (TODO: to be revised) Ensure that they are not negative
        if self.ixy_w < 0 and np.abs(self.ixy_w / self.hw) < 1e-5:
            self.ixy_w = 0.
        if self.ixz_w < 0 and np.abs(self.ixz_w / self.hw) < 1e-5:
            self.ixz_w = 0.
        if self.iyz_w < 0 and np.abs(self.iyz_w / self.hw) < 1e-5:
            self.iyz_w = 0.
        if self.hx_w < 0 and np.abs(self.hx_w / self.hw) < 1e-5:
            self.hx_w = 0.
        if self.hy_w < 0 and np.abs(self.hy_w / self.hw) < 1e-5:
            self.hy_w = 0.

        # Compute MIIT
        self.ii = self.hxyw + self.hyzw + self.hxzw + self.hw - self.hxw - self.hyw - self.hzw - self.hxyzw
        self.itot = self.ii + self.ixz_w + self.iyz_w

        # Compute R(Z;X,Y|W)
        self.rmmi    = np.min([self.ixz_w, self.iyz_w])                # RMMIc
        self.isource = self.ixy_w / np.min([self.hxw, self.hyw])       # Isc
        # self.isource = self.ixy_w / np.min([self.hx_w, self.hy_w])       # Isc
        # self.isource = 0.       # Isc
        self.rmin    = -self.ii if self.ii < 0 else 0                  # Rminc
        self.r       = self.rmin + self.isource*(self.rmmi-self.rmin)  # Rc

        # Compute S(Z;X,Y|W), U(Z;X|W) and U(Z;Y|W)
        self.s = self.r + self.ii       # Sc
        self.uxz = self.ixz_w - self.r  # U(X;Z|W)
        self.uyz = self.iyz_w - self.r  # U(Y;Z|W)

    def __computeInfoMD2(self, pdfs, pdfs1, pdfs2):
        '''
        The function is aimed to compute the momentary interaction information at two causal paths and
        its corresponding momentary inforamtion partitioning, however, the
        Compute
                II(X;Z;Y|W) = I(X,Y;Z|W) - I(X;Z|W1) - I(Y;Z|W2)
                Isc = I(X;Y|W) / min[H(X|W), H(Y|W)]
                RMMIc = min[I(X;Z|W1), I(Y;Z|W2)]
                Rminc = 0 if II > 0 else -II
                Rc = Rminc + Isc*(RMMIc - Rminc)
                Sc = II + Rc
                Uxc = I(X;Z|W1) - Rc
                Uyc = I(Y:Z|W2) - Rc
        Input:
        pdfs --  a numpy array with shape (nx, ny, nz, nw1, nw2, nw3,...)
        Output: NoneType
        '''

        ################################################
        # Compute I(X,Y;Z|W), I(X;Y|W), H(X|W), H(Y|W) #
        ################################################
        shapes = pdfs.shape
        ndims  = len(shapes)
        nx, ny, nz, nws = shapes[0], shapes[1], shapes[2], shapes[3:]
        wpdfs = np.sum(pdfs, axis=(0,1,2))   # p(w)
        xwpdfs, ywpdfs, zwpdfs = np.sum(pdfs, axis=(1,2)), np.sum(pdfs, axis=(0,2)), np.sum(pdfs, axis=(0,1))  # p(x,w), p(y,w), p(z,w)
        xywpdfs, yzwpdfs, xzwpdfs = np.sum(pdfs, axis=(2)), np.sum(pdfs, axis=(0)), np.sum(pdfs, axis=(1))  # p(x,y,w), p(y,z,w), p(x,z,w)

        # Compute all the entropies
        hw    = computeEntropy(wpdfs.flatten(), base=self.base)    # H(W)
        hxw   = computeEntropy(xwpdfs.flatten(), base=self.base)   # H(X,W)
        hyw   = computeEntropy(ywpdfs.flatten(), base=self.base)   # H(Y,W)
        hzw   = computeEntropy(zwpdfs.flatten(), base=self.base)   # H(Z,W)
        hxyw  = computeEntropy(xywpdfs.flatten(), base=self.base)  # H(X,Y,W)
        hxyzw = computeEntropy(pdfs.flatten(), base=self.base)     # H(X,Y,Z,W)

        # Compute I(X,Y;Z|W), I(X;Y|W), H(X|W), H(Y|W)
        self.hx_w = hxw - hw  # H(X|W)
        self.hy_w = hyw - hw  # H(Y|W)
        self.ixy_w = hxw + hyw - hw - hxyw  # I(X;Y|W)
        self.itot = hxyw + hzw - hxyzw -hw # I(X,Y;Z|W)

        ## (TODO: to be revised) Ensure that they are not negative
        if self.ixy_w < 0 and np.abs(self.ixy_w / self.hw) < 1e-5:
            self.ixy_w = 0.
        if self.itot < 0 and np.abs(self.itot / self.hw) < 1e-5:
            self.itot = 0.
        if self.hx_w < 0 and np.abs(self.hx_w / self.hw) < 1e-5:
            self.hx_w = 0.
        if self.hy_w < 0 and np.abs(self.hy_w / self.hw) < 1e-5:
            self.hy_w = 0.

        #####################
        # Compute I(X;Z|W1) #
        #####################
        shapes1 = pdfs1.shape
        ndims1  = len(shapes1)
        nx1, ny1, nz1, nws1 = shapes1[0], shapes1[1], shapes1[2], shapes1[3:]
        wpdfs1 = np.sum(pdfs1, axis=(0,1,2))   # p(w)
        xwpdfs1, ywpdfs1, zwpdfs1 = np.sum(pdfs1, axis=(1,2)), np.sum(pdfs1, axis=(0,2)), np.sum(pdfs1, axis=(0,1))  # p(x,w), p(y,w), p(z,w)
        xywpdfs1, yzwpdfs1, xzwpdfs1 = np.sum(pdfs1, axis=(2)), np.sum(pdfs1, axis=(0)), np.sum(pdfs1, axis=(1))  # p(x,y,w), p(y,z,w), p(x,z,w)

        # Compute all the entropies
        hw    = computeEntropy(wpdfs1.flatten(), base=self.base)    # H(W)
        hxw   = computeEntropy(xwpdfs1.flatten(), base=self.base)   # H(X,W)
        hzw   = computeEntropy(zwpdfs1.flatten(), base=self.base)   # H(Z,W)
        hxzw  = computeEntropy(xzwpdfs1.flatten(), base=self.base)  # H(X,Z,W)

        # Compute I(X;Z|W1)
        self.ixz_w = hxw + hzw - hw - hxzw  # I(X;Z|W1)

        ## (TODO: to be revised) Ensure that they are not negative
        if self.ixz_w < 0 and np.abs(self.ixz_w / self.hw) < 1e-5:
            self.ixz_w = 0.

        #####################
        # Compute I(Y;Z|W2) #
        #####################
        shapes2 = pdfs2.shape
        ndims2  = len(shapes2)
        nx2, ny2, nz2, nws2 = shapes2[0], shapes2[1], shapes2[2], shapes2[3:]
        wpdfs2 = np.sum(pdfs2, axis=(0,1,2))   # p(w)
        xwpdfs2, ywpdfs2, zwpdfs2 = np.sum(pdfs2, axis=(1,2)), np.sum(pdfs2, axis=(0,2)), np.sum(pdfs2, axis=(0,1))  # p(x,w), p(y,w), p(z,w)
        xywpdfs2, yzwpdfs2, xzwpdfs2 = np.sum(pdfs2, axis=(2)), np.sum(pdfs2, axis=(0)), np.sum(pdfs2, axis=(1))  # p(x,y,w), p(y,z,w), p(x,z,w)

        # Compute all the entropies
        hw    = computeEntropy(wpdfs2.flatten(), base=self.base)    # H(W)
        hyw   = computeEntropy(ywpdfs2.flatten(), base=self.base)   # H(Y,W)
        hzw   = computeEntropy(zwpdfs2.flatten(), base=self.base)   # H(Z,W)
        hyzw  = computeEntropy(yzwpdfs2.flatten(), base=self.base)  # H(Y,Z,W)

        # Compute I(Y;Z|W2)
        self.iyz_w = hyw + hzw - hw - hxzw  # I(Y;Z|W2)

        ## (TODO: to be revised) Ensure that they are not negative
        if self.iyz_w < 0 and np.abs(self.iyz_w / self.hw) < 1e-5:
            self.iyz_w = 0.

        ################
        # Compute MPID #
        ################
        self.ii = self.itot - self.ixz_w - self.iyz_w

        # Compute R(Z;X,Y|W)
        self.rmmi    = np.min([self.ixz_w, self.iyz_w])                # RMMIc
        self.isource = self.ixy_w / np.min([self.hx_w, self.hy_w])       # Isc
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
        # else:
        #     self.allInfo = pd.Series([self.ii, self.itot, self.r, self.s, self.uxz, self.uyz,
        #                               self.rmmi, self.isource, self.rmin],
        #                              index=['MIIT', 'Itotal', 'Rc', 'Sc', 'Uxc', 'Uyc',
        #                                     'RMMIc', 'Isc', 'Rminc'])

##################
# Help functions #
##################
def equal(a, b, e=1e-10):
    '''Check whether the two numbers are equal'''
    return np.abs(a - b) < e

def computeEntropy(pdfs, base=2):
    '''Compute the entropy H(X).'''
    # Calculate the log of pdf
    pdfs_log = np.ma.log(pdfs)
    pdfs_log = pdfs_log.filled(0) / np.log(base)

    # Calculate H(X)
    return -np.sum(pdfs*pdfs_log)


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
