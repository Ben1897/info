"""
A class for calculating the statistical information.

1D: H(X)
2D: H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
3D: H(X1), H(Y), H(X2), I(X1;Y), I(X1;X2), I(X2;Y), T(Y->X), II, I(X1,Y;X2), R, S, U1, U2

class info()
  __init__()
  __computeInfo1D()
  __computeInfo1D_conditioned()
  __computeInfo2D()
  __computeInfo2D_conditioned()
  __computeInfo3D()
  __computeInfo3D_specific()
  __computeInfo3D_specific_wrong()
  __computeInfo3D_conditioned()
  __computeInfoMD2()
  __assemble()

equal()
computeEntropy()
computeConditionalInfo()
computeMutualInfo()
computeConditionalMutualInformation()

Ref:
Allison's SUR paper

"""

import numpy as np
import pandas as pd
from ..utils.pdf_computer import pdf_computer
# from scipy.stats import entropy


class info(object):

    def __init__(self, ndim, data, approach='kde_c', bandwidth='silverman', kernel='gaussian',
                 base=2, conditioned=False, specific=False, averaged=True, onlycmi=False):
        '''
        Input:
        ndim        -- the number of dimension to be computed [int]
        data        -- the data [numpy array with shape (npoints, ndim)]
        approach    -- the code for computing PDF by using KDE
        kernel      -- the kernel type [string]
        bandwith    -- the band with of the kernel [string or float]
        base        -- the logrithmatic base (the default is 2) [float/int]
        conditioned -- whether including conditions [bool]
        specific    -- whether calculating the specific PID [bool]
        averaged    -- whether computing the average value of each info bit or using the traditional discrete formula [averaged]
        '''
        self.base        = base
        self.conditioned = conditioned
        self.specific    = specific
        self.averaged    = averaged

        # Check the dimension of the data
        if len(data.shape) > 2:
            raise Exception('The dimension of the data matrix is not (npts, ndim)!')
        if ndim == 1 and len(data.shape) == 1:
            data = data[:,np.newaxis]
        npts, ndimdata = data.shape
        if ndim != ndimdata and not conditioned:
            raise Exception('The dimension of the variables is %d, not %d!' % (ndimdata, ndim))
        elif ndim >= ndimdata and conditioned:
            raise Exception('The dimension of the variables should be larger than %d, not %d!' % (ndimdata, ndim))
        self.npts = npts
        self.ndim = ndim
        self.data = data

        # Initiate the PDF computer
        self.computer = pdf_computer(approach=approach, bandwidth=bandwidth, kernel=kernel)

        if not onlycmi:
            # 1D
            if self.ndim == 1 and not conditioned:
                self.__computeInfo1D()
            elif self.ndim == 1 and conditioned:
                self.__computeInfo1D_conditioned()

            # 2D
            if self.ndim == 2 and not conditioned:
                self.__computeInfo2D()
            elif self.ndim == 2 and conditioned:
                self.__computeInfo2D_conditioned()

            # 3D
            if self.ndim == 3 and not conditioned:
                self.__computeInfo3D()
            elif self.ndim == 3 and conditioned:
                self.__computeInfo3D_conditioned()

            # Assemble all the information values into a Pandas series format
            self.__assemble()

    def __computeInfo1D(self):
        '''
        Compute H(X)
        Input:
        Output: NoneType
        '''
        base     = self.base
        data     = self.data
        computer = self.computer
        averaged = self.averaged

        # Compute the pdfs
        _, pdfs = computer(data)

        # Compute information metrics
        self.hx = computeEntropy(pdfs, base=base, averaged=averaged)

    def __computeInfo1D_conditioned(self):
        '''
        Compute H(X|W)
        '''
        base     = self.base
        data     = self.data
        computer = self.computer
        averaged = self.averaged

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs = computer.computePDF(data[:,[0]])
        _, wpdfs = computer.computePDF(data[:,1:])

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs, base=base, averaged=averaged)    # H(W)
        self.hx    = computeEntropy(xpdfs, base=base, averaged=averaged)    # H(X)
        self.hxw   = computeEntropy(pdfs, base=base, averaged=averaged)     # H(X,W)
        self.hx_w  = self.hxw - self.hw                                     # H(X|W)

    def __computeInfo2D(self):
        '''
        Compute H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
        Input:
        pdfs --  a numpy array with shape (nx, ny)
        Output: NoneType
        '''
        base     = self.base
        data     = self.data
        computer = self.computer
        averaged = self.averaged

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs = computer.computePDF(data[:,[0]])
        _, ypdfs = computer.computePDF(data[:,[1]])

        # Compute H(X), H(Y) and H(X,Y)
        # print xpdfs
        self.hx  = computeEntropy(xpdfs, base=base, averaged=averaged)  # H(X)
        self.hy  = computeEntropy(ypdfs, base=base, averaged=averaged)  # H(Y)
        self.hxy = computeEntropy(pdfs, base=base, averaged=averaged)   # H(X,Y)
        self.hy_x = self.hxy - self.hx                                  # H(Y|X)
        self.hx_y = self.hxy - self.hy                                  # H(X|Y)
        self.ixy  = self.hx + self.hy - self.hxy                        # I(X;Y)

    def __computeInfo2D_conditioned(self):
        '''
        Compute H(X|W), H(Y|W), H(X,Y|W), I(X,Y|W)
        '''
        base       = self.base
        data       = self.data
        computer   = self.computer
        averaged   = self.averaged
        npts, ndim = data.shape

        # Compute the pdfs
        _, pdfs   = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,[0]])
        _, ypdfs  = computer.computePDF(data[:,[1]])
        _, wpdfs  = computer.computePDF(data[:,2:])
        _, xypdfs = computer.computePDF(data[:,[0,1]])
        _, xwpdfs = computer.computePDF(data[:,[0]+range(2,ndim)])
        _, ywpdfs = computer.computePDF(data[:,[1]+range(2,ndim)])

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs, base=base, averaged=averaged)    # H(W)
        self.hx    = computeEntropy(xpdfs, base=base, averaged=averaged)    # H(X)
        self.hy    = computeEntropy(ypdfs, base=base, averaged=averaged)    # H(Y)
        self.hxy   = computeEntropy(xypdfs, base=base, averaged=averaged)   # H(X,Y)
        self.hxw   = computeEntropy(xwpdfs, base=base, averaged=averaged)   # H(X,W)
        self.hyw   = computeEntropy(ywpdfs, base=base, averaged=averaged)   # H(Y,W)
        self.hxyw  = computeEntropy(pdfs, base=base, averaged=averaged)     # H(X,Y,W)
        self.hx_w  = self.hxw - self.hw                                     # H(X|W)
        self.hy_w  = self.hyw - self.hw                                     # H(Y|W)
        self.hx_y  = self.hxy - self.hy                                     # H(X|Y)
        self.hy_x  = self.hxy - self.hx                                     # H(Y|X)

        # Compute all the conditional mutual information
        self.ixy   = self.hx + self.hy - self.hxy                           # I(X;Y)
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw              # I(X;Y|W)

    def __computeInfo3D(self):
        '''
        Compute H(X), H(Y), H(Z), I(Y;Z), I(X;Z), I(X;Y), I(Y,Z|X), I(X,Z|Y), II,
                I(X,Y;Z), R, S, U1, U2
        Here, X --> X2, Z --> Xtar, Y --> X1 in Allison's TIPNets manuscript.
        Input:
        pdfs --  a numpy array with shape (nx, ny, nz)
        Output: NoneType
        '''
        base       = self.base
        data       = self.data
        computer   = self.computer
        averaged   = self.averaged
        npts, ndim = data.shape

        # Compute the pdfs
        _, pdfs   = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,[0]])
        _, ypdfs  = computer.computePDF(data[:,[1]])
        _, zpdfs  = computer.computePDF(data[:,[2]])
        _, xypdfs = computer.computePDF(data[:,[0,1]])
        _, xzpdfs = computer.computePDF(data[:,[0,2]])
        _, yzpdfs = computer.computePDF(data[:,[1,2]])

        # Compute H(X), H(Y) and H(Z)
        self.hx   = computeEntropy(xpdfs, base=base, averaged=averaged)   # H(X)
        self.hy   = computeEntropy(ypdfs, base=base, averaged=averaged)   # H(Y)
        self.hz   = computeEntropy(zpdfs, base=base, averaged=averaged)   # H(Z)
        self.hxy  = computeEntropy(xypdfs, base=base, averaged=averaged)  # H(X,Y)
        self.hyz  = computeEntropy(yzpdfs, base=base, averaged=averaged)  # H(Y,Z)
        self.hxz  = computeEntropy(xzpdfs, base=base, averaged=averaged)  # H(X,Z)
        self.hxyz = computeEntropy(pdfs, base=base, averaged=averaged)    # H(X,Y,Z)

        # Compute I(X;Z), I(Y;Z) and I(X;Y)
        self.ixy = self.hx + self.hy - self.hxy                           # I(X;Z)
        self.ixz = self.hx + self.hz - self.hxz                           # I(Y;Z)
        self.iyz = self.hy + self.hz - self.hyz                           # I(X;Y)

        # Compute II (= I(X;Y;Z))
        self.itot = self.hxy + self.hz - self.hxyz                        # I(X,Y;Z)
        self.ii   = self.itot - self.ixz - self.iyz                       # interaction information

        # Compute R(Z;X,Y)
        self.rmmi    = np.min([self.ixz, self.iyz])                       # RMMI (Eq.(7) in Allison)
        self.isource = self.ixy / np.min([self.hx, self.hy])              # Is (Eq.(9) in Allison)
        self.rmin    = -self.ii if self.ii < 0 else 0                     # Rmin (Eq.(10) in Allison)
        self.r       = self.rmin + self.isource*(self.rmmi-self.rmin)     # Rs (Eq.(11) in Allison)
        # self.r       = self.rmmi

        # Compute S(Z;X,Y), U(Z;X) and U(Z;Y)
        self.s = self.r + self.ii     # S (II = S - R)
        self.uxz = self.ixz - self.r  # U(X;Z) (Eq.(4) in Allison)
        self.uyz = self.iyz - self.r  # U(Y;Z) (Eq.(5) in Allison)

    def __computeInfo3D_conditioned(self):
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
        base       = self.base
        data       = self.data
        computer   = self.computer
        averaged   = self.averaged
        npts, ndim = data.shape

        # Compute the pdfs
        _, pdfs    = computer.computePDF(data)
        _, xpdfs   = computer.computePDF(data[:,[0]])
        _, ypdfs   = computer.computePDF(data[:,[1]])
        _, zpdfs   = computer.computePDF(data[:,[2]])
        _, wpdfs   = computer.computePDF(data[:,3:])
        _, xypdfs  = computer.computePDF(data[:,[0,1]])
        _, xzpdfs  = computer.computePDF(data[:,[0,2]])
        _, yzpdfs  = computer.computePDF(data[:,[1,2]])
        _, xwpdfs  = computer.computePDF(data[:,[0]+range(3,ndim)])
        _, ywpdfs  = computer.computePDF(data[:,[1]+range(3,ndim)])
        _, zwpdfs  = computer.computePDF(data[:,[2]+range(3,ndim)])
        _, xywpdfs = computer.computePDF(data[:,[0,1]+range(3,ndim)])
        _, yzwpdfs = computer.computePDF(data[:,[1,2]+range(3,ndim)])
        _, xzwpdfs = computer.computePDF(data[:,[0,2]+range(3,ndim)])

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs, base=base, averaged=averaged)    # H(W)
        self.hx    = computeEntropy(xpdfs, base=base, averaged=averaged)    # H(X)
        self.hy    = computeEntropy(ypdfs, base=base, averaged=averaged)    # H(Y)
        self.hz    = computeEntropy(zpdfs, base=base, averaged=averaged)    # H(Z)
        self.hxw   = computeEntropy(xwpdfs, base=base, averaged=averaged)   # H(X,W)
        self.hyw   = computeEntropy(ywpdfs, base=base, averaged=averaged)   # H(Y,W)
        self.hzw   = computeEntropy(zwpdfs, base=base, averaged=averaged)   # H(Z,W)
        self.hxyw  = computeEntropy(xywpdfs, base=base, averaged=averaged)  # H(X,Y,W)
        self.hyzw  = computeEntropy(yzwpdfs, base=base, averaged=averaged)  # H(Y,Z,W)
        self.hxzw  = computeEntropy(xzwpdfs, base=base, averaged=averaged)  # H(X,Z,W)
        self.hxyzw = computeEntropy(pdfs, base=base, averaged=averaged)     # H(X,Y,Z,W)
        self.hx_w  = self.hxw - self.hw                                     # H(X|W)
        self.hy_w  = self.hyw - self.hw                                     # H(Y|W)

        # Compute all the conditional mutual information
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw              # I(X;Y|W)
        self.ixz_w = self.hxw + self.hzw - self.hw - self.hxzw              # I(X;Z|W)
        self.iyz_w = self.hyw + self.hzw - self.hw - self.hyzw              # I(Y;Z|W)

        ## (TODO: to be revised) Ensure that they are nonnegative
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
        self.rmmi    = np.min([self.ixz_w, self.iyz_w])                     # RMMIc
        # self.isource = self.ixy_w / np.min([self.hxw, self.hyw])            # Isc
        self.isource = self.ixy_w / np.min([self.hx_w, self.hy_w])        # Isc
        self.rmin    = -self.ii if self.ii < 0 else 0                       # Rminc
        self.r       = self.rmin + self.isource*(self.rmmi-self.rmin)       # Rc

        # Compute S(Z;X,Y|W), U(Z;X|W) and U(Z;Y|W)
        self.s = self.r + self.ii                                           # Sc
        self.uxz = self.ixz_w - self.r                                      # U(X;Z|W)
        self.uyz = self.iyz_w - self.r                                      # U(Y;Z|W)

    def computeInfo2D_multiple_conditioned(self, xlastind, ylastind):
        '''
        Compute H(Xset|W), H(Yset|W), H(Xset,Yset|W), I(Xset,Yset|W)
        used for computing the accumulated information transfer (AIT)
        '''
        base       = self.base
        data       = self.data
        computer   = self.computer
        averaged   = self.averaged
        npts, ndim = data.shape

        if xlastind > ylastind:
            raise Exception("xlastind %d is larger than ylastind %d" % (xlastind, ylastind))
        if xlastind > ndim-1:
            raise Exception("xlastind %d is larger than the maximum dimension" % xlastind)
        if ylastind > ndim-1:
            raise Exception("ylastind %d is larger than the maximum dimension" % ylastind)

        # Compute the pdfs
        _, pdfs   = computer.computePDF(data)
        # _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
        # _, ypdfs  = computer.computePDF(data[:,range(xlastind,ylastind)])
        _, wpdfs  = computer.computePDF(data[:,range(ylastind,ndim)])
        # _, xypdfs = computer.computePDF(data[:,range(0,ylastind)])
        _, xwpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,ndim)])
        _, ywpdfs = computer.computePDF(data[:,range(xlastind,ndim)])

        # Compute all the conditional mutual information
        # Calculate the log of pdf
        pdfs_log, wpdfs_log    = np.ma.log(pdfs), np.ma.log(wpdfs)
        xwpdfs_log, ywpdfs_log = np.ma.log(xwpdfs), np.ma.log(ywpdfs)
        pdfs_log = (pdfs_log.filled(0) + wpdfs_log.filled(0) - xwpdfs_log.filled(0) - ywpdfs_log.filled(0)) / np.log(base)

        # Normalize the joint PDF
        pdfs_norm = pdfs / pdfs.sum()

        # The conditional probability
        return np.sum(pdfs_norm*pdfs_log)

    def __assemble(self):
        '''
        Assemble all the information values into a Pandas series format
        Output: NoneType
        '''
        if self.ndim == 1 and not self.conditioned:
            self.allInfo = pd.Series(self.hx, index=['H(X)'], name='ordinary')

        elif self.ndim == 1 and self.conditioned:
            self.allInfo = pd.Series([self.hx, self.hx_w], index=['H(X)', 'H(X|W)'], name='ordinary')

        elif self.ndim == 2 and not self.conditioned:
            self.allInfo = pd.Series([self.hx, self.hy, self.hx_y, self.hy_x, self.ixy],
                                     index=['H(X)', 'H(Y)', 'H(X|Y)', 'H(Y|X)', 'I(X;Y)'],
                                     name='ordinary')

        elif self.ndim == 2 and self.conditioned:
            self.allInfo = pd.Series([self.hx, self.hx_y, self.hxyw-self.hyw, self.ixy_w],
                                     index=['H(X)', 'H(X|Y)', 'H(X|Y,W)', 'I(X;Y|W)'],
                                     name='ordinary')

        elif self.ndim == 3 and not self.conditioned:
            self.allInfo = pd.Series([self.ixz, self.iyz, self.itot, self.ii, self.r, self.s, self.uxz, self.uyz, self.rmin, self.isource, self.rmmi],
                                     index=['I(X;Z)', 'I(Y;Z)', 'I(X,Y;Z)', 'II', 'R(Z;Y,X)', 'S(Z;Y,X)', 'U(Z,X)', 'U(Z,Y)', 'Rmin', 'Isource', 'RMMI'],
                                     name='ordinary')

        elif self.ndim == 3 and self.conditioned:
            self.allInfo = pd.Series([self.ixz_w, self.iyz_w, self.itot, self.ii, self.r, self.s, self.uxz, self.uyz, self.rmin, self.isource, self.rmmi],
                                     index=['I(X;Z|W)', 'I(Y;Z|W)', 'I(X,Y;Z|W)', 'II', 'R(Z;Y,X|W)', 'S(Z;Y,X|W)', 'U(Z,X|W)', 'U(Z,Y|W)', 'Rmin', 'Isource', 'RMMI'],
                                     name='ordinary')

    def normalizeinfo(self):
        """
        Normalize the calculated information metrics in terms of both percentage and magnitude.

        Note that for 2D and 3D, the magnitude-based normalization emphasizes the amount of information transfer
        given by the source(s) to the target Y compared with the information of Y itself. Therefore, the scaling base
        is H(Z). Meanwhile, the percentage-based normalization scales the metrics in terms of the joint uncerntainty
        of the source(s) and the target Z with the condition considered. Therefore, the scaling base is the joint entropy which
        is conditioned if the condition W exists.

        For 1D, (i.e., X with the condition W), the scaling bases are:
            unconditioned:              Hmax(X) = log(Nx)/log(base)
            conditioned (percentage):   H(X)
            conditioned (magnitude):    Hmax(X,W) = log(Nx*Nw)/log(base)
        For 2D, (i.e., X (source) and Y (target) with the condition W),the scaling bases are:
            unconditioned (percentage): H(X,Y)
            unconditioned (magnitude):  H(Y)
            conditioned (percentage):   H(X,Y|W)
            conditioned (magnitude):    H(Y)
        For 3D, (i.e., X, Y (sources) and Z (target) with the condition W),the scaling bases are:
            unconditioned (percentage): H(X,Y,Z)
            unconditioned (magnitude):  H(Z)
            conditioned (percentage):   H(X,Y,Z|W)
            conditioned (magnitude):    H(Z)
        """
        base, npts = self.base, self.npts

        # Check whether it is the specific information and return None if yes
        if self.specific:
            print "The specific information is not considered for normalization yet!"
            return

        # 1D - unconditioned
        # unconditioned: Hmax(X) = log(Nx)/log(base)
        if self.ndim == 1 and not self.conditioned:
            nx           = npts
            scalingbase  = np.log(nx) / np.log(base)
            self.hx_norm = self.hx / scalingbase
            # assemble it to pandas series
            norm_df = pd.Series(self.hx_norm, index=['H(X)'], name='norm')

        # 1D - conditioned
        # conditioned (percentage): H(X)
        # conditioned (magnitude):  Hmax(X,W) = log(Nx*Nw)/log(base)
        if self.ndim == 1 and self.conditioned:
            nx, nw          = npts, npts
            # percentage
            scalingbase_p   = self.hx
            self.hx_w_normp = self.hx_w / scalingbase_p
            # magnitude
            scalingbase_m   = np.log(nx*nw) / np.log(base)
            self.hx_w_normm = self.hx_w / scalingbase_m
            # assemble them to pandas series
            norm_p_df = pd.Series(self.hx_w_normp, index=['H(X|W)'], name='norm_p')
            norm_m_df = pd.Series(self.hx_w_normm, index=['H(X|W)'], name='norm_m')

        # 2D - unconditioned
        # X: source, Y: target
        # unconditioned (percentage): H(X,Y)
        # unconditioned (magnitude):  H(Y)
        if self.ndim == 2 and not self.conditioned:
            # percentage
            scalingbase_p   = self.hx_y + self.hy   # H(X,Y)
            self.hx_y_normp = self.hx_y / scalingbase_p
            self.ixy_normp  = self.ixy / scalingbase_p
            # magnitude
            scalingbase_m   = self.hy               # H(Y)
            self.hx_y_normm = self.hx_y / scalingbase_m
            self.ixy_normm  = self.ixy / scalingbase_m
            # assemble them to pandas series
            norm_p_df = pd.Series([self.hx_y_normp, self.ixy_normp],
                                  index=['H(X|Y)', 'I(X;Y)'], name='norm_p')
            norm_m_df = pd.Series([self.hx_y_normm, self.ixy_normm],
                                  index=['H(X|Y)', 'I(X;Y)'], name='norm_m')

        # 2D - conditioned
        # X: source, Y: target, W: condition
        # conditioned (percentage): H(X,Y|W)
        # conditioned (magnitude):  H(Y)
        if self.ndim == 2 and self.conditioned:
            # percentage
            scalingbase_p    = self.hxyw - self.hw   # H(X,Y|W)
            self.hx_yw_normp = (self.hxyw - self.hyw) / scalingbase_p
            self.ixy_w_normp = self.ixy_w / scalingbase_p
            # magnitude
            scalingbase_m    = self.hy               # H(Y)
            self.hx_yw_normm = (self.hxyw - self.hyw) / scalingbase_m
            self.ixy_w_normm = self.ixy_w / scalingbase_m
            # assemble them to pandas series
            norm_p_df = pd.Series([self.hx_yw_normp, self.ixy_w_normp],
                                  index=['H(X|Y,W)', 'I(X;Y|W)'], name='norm_p')
            norm_m_df = pd.Series([self.hx_yw_normm, self.ixy_w_normm],
                                  index=['H(X|Y,W)', 'I(X;Y|W)'], name='norm_m')

        # 3D - unconditioned
        # X, Y: sourceS, Z: target
        # unconditioned (percentage): H(X,Y,Z)
        # unconditioned (magnitude):  H(Z)
        if self.ndim == 3 and not self.conditioned:
            # percentage
            scalingbase_p   = self.hxyz               # H(X,Y,Z)
            self.ixz_normp  = self.ixz / scalingbase_p
            self.iyz_normp  = self.iyz / scalingbase_p
            self.itot_normp = self.itot / scalingbase_p
            self.ii_normp   = self.ii / scalingbase_p
            self.r_normp    = self.r / scalingbase_p
            self.s_normp    = self.s/ scalingbase_p
            self.uxz_normp  = self.uxz / scalingbase_p
            self.uyz_normp  = self.uyz / scalingbase_p
            self.rmin_normp = self.rmin / scalingbase_p
            self.isource_normp = self.isource / scalingbase_p
            self.rmmi_normp = self.rmmi / scalingbase_p
            # magnitude
            scalingbase_m   = self.hz                 # H(Z)
            self.ixz_normm  = self.ixz / scalingbase_m
            self.iyz_normm  = self.iyz / scalingbase_m
            self.itot_normm = self.itot / scalingbase_m
            self.ii_normm   = self.ii / scalingbase_m
            self.r_normm    = self.r / scalingbase_m
            self.s_normm    = self.s/ scalingbase_m
            self.uxz_normm  = self.uxz / scalingbase_m
            self.uyz_normm  = self.uyz / scalingbase_m
            self.rmin_normm = self.rmin / scalingbase_m
            self.isource_normm = self.isource / scalingbase_m
            self.rmmi_normm = self.rmmi / scalingbase_m
            # assemble them to pandas series
            norm_p_df = pd.Series([self.ixz_normp, self.iyz_normp, self.itot_normp, self.ii_normp, self.r_normp, self.s_normp,
                                   self.uxz_normp, self.uyz_normp, self.rmin_normp, self.isource_normp, self.rmmi_normp],
                                  index=['I(X;Z)', 'I(Y;Z)', 'I(X,Y;Z)', 'II', 'R(Z;Y,X)', 'S(Z;Y,X)', 'U(Z,X)', 'U(Z,Y)', 'Rmin', 'Isource', 'RMMI'],
                                  name='norm_p')
            norm_m_df = pd.Series([self.ixz_normm, self.iyz_normm, self.itot_normm, self.ii_normm, self.r_normm, self.s_normm,
                                   self.uxz_normm, self.uyz_normm, self.rmin_normm, self.isource_normm, self.rmmi_normm],
                                  index=['I(X;Z)', 'I(Y;Z)', 'I(X,Y;Z)', 'II', 'R(Z;Y,X)', 'S(Z;Y,X)', 'U(Z,X)', 'U(Z,Y)', 'Rmin', 'Isource', 'RMMI'],
                                  name='norm_m')

        # 3D - conditioned
        # X, Y: sourceS, Z: target, W: condition
        # conditioned (percentage):   H(X,Y,Z|W)
        # conditioned (magnitude):    H(Z)
        if self.ndim == 3 and self.conditioned:
            # percentage
            scalingbase_p    = self.hxyzw - self.hw    # H(X,Y,Z|W)
            self.ixz_w_normp = self.ixz_w / scalingbase_p
            self.iyz_w_normp = self.iyz_w / scalingbase_p
            self.itot_normp  = self.itot / scalingbase_p
            self.ii_normp    = self.ii / scalingbase_p
            self.r_normp     = self.r / scalingbase_p
            self.s_normp     = self.s/ scalingbase_p
            self.uxz_normp   = self.uxz / scalingbase_p
            self.uyz_normp   = self.uyz / scalingbase_p
            self.rmin_normp = self.rmin / scalingbase_p
            self.isource_normp = self.isource / scalingbase_p
            self.rmmi_normp = self.rmmi / scalingbase_p
            # magnitude
            scalingbase_m    = self.hz                 # H(Z)
            self.ixz_w_normm = self.ixz_w / scalingbase_m
            self.iyz_w_normm = self.iyz_w / scalingbase_m
            self.itot_normm  = self.itot / scalingbase_m
            self.ii_normm    = self.ii / scalingbase_m
            self.r_normm     = self.r / scalingbase_m
            self.s_normm     = self.s/ scalingbase_m
            self.uxz_normm   = self.uxz / scalingbase_m
            self.uyz_normm   = self.uyz / scalingbase_m
            self.rmin_normm = self.rmin / scalingbase_m
            self.isource_normm = self.isource / scalingbase_m
            self.rmmi_normm = self.rmmi / scalingbase_m
            # assemble them to pandas series
            norm_p_df = pd.Series([self.ixz_w_normp, self.iyz_w_normp, self.itot_normp, self.ii_normp, self.r_normp, self.s_normp,
                                   self.uxz_normp, self.uyz_normp, self.rmin_normp, self.isource_normp, self.rmmi_normp],
                                  index=['I(X;Z|W)', 'I(Y;Z|W)', 'I(X,Y;Z|W)', 'II', 'R(Z;Y,X|W)', 'S(Z;Y,X|W)', 'U(Z,X|W)', 'U(Z,Y|W)', 'Rmin', 'Isource', 'RMMI'],
                                  name='norm_p')
            norm_m_df = pd.Series([self.ixz_w_normm, self.iyz_w_normm, self.itot_normm, self.ii_normm, self.r_normm, self.s_normm,
                                   self.uxz_normm, self.uyz_normm, self.rmin_normm, self.isource_normm, self.rmmi_normm],
                                  index=['I(X;Z|W)', 'I(Y;Z|W)', 'I(X,Y;Z|W)', 'II', 'R(Z;Y,X|W)', 'S(Z;Y,X|W)', 'U(Z,X|W)', 'U(Z,Y|W)', 'Rmin', 'Isource', 'RMMI'],
                                  name='norm_m')

        # Assemble all the information metrics
        if self.ndim == 1 and not self.conditioned:
            self.allInfo = pd.concat([self.allInfo, norm_df], axis=1)
        else:
            self.allInfo = pd.concat([self.allInfo, norm_p_df, norm_m_df], axis=1)


##################
# Help functions #
##################
def equal(a, b, e=1e-10):
    '''Check whether the two numbers are equal'''
    return np.abs(a - b) < e

def computeEntropy(pdfs, base=2, averaged=True):
    '''Compute the entropy H(X).'''
    # Calculate the log of pdf
    pdfs_log = np.ma.log(pdfs)
    pdfs_log = pdfs_log.filled(0) / np.log(base)

    # Calculate H(X)
    if averaged:
        return -np.mean(pdfs_log)
    elif not averaged:
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
