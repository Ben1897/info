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
  __comptueInfo2D_multivariate_conditioned()
  __computeInfo3D()
  __computeInfo3D_conditioned()
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
from ..utils.knntoolkit import knn_cuda, knn_scipy
# from scipy.stats import entropy

kde_approaches = ['kde_c', 'kde_cuda', 'kde_cuda_general']
knn_approaches = ['knn_cuda', 'knn_scipy']

class info(object):

    def __init__(self, case, data, approach='kde_c', bandwidth='silverman', kernel='gaussian', k=10,
                 base=2, conditioned=False, specific=False, averaged=True, xyindex=None):
        '''
        Input:
        case        -- the number of dimension to be computed [int]
        data        -- the data [numpy array with shape (npoints, ndim)]
        approach    -- the code for computing PDF by using KDE
        kernel      -- the kernel type [string]
        bandwith    -- the band with of the kernel [string or float]
        k           -- the number of the nearest neighbor used in KNN method [int]
        base        -- the logrithmatic base (the default is 2) [float/int]
        conditioned -- whether including conditions [bool]
        specific    -- whether calculating the specific PID [bool]
        averaged    -- whether computing the average value of each info bit or using the traditional discrete formula [averaged]
        xyindex     -- a list of index indicating the position of the involved variable set, used for computeInfo*D_multivariate*
                       1D: [xlastind], 2D: [xlastind, ylastind], 3D: [xlastind,ylastind,zlastind]
                       note that xlastind < ylastind < zlastind <= len(pdfs.shape)
                       if None, used for computeInfo*D*
        '''
        self.base        = base
        self.conditioned = conditioned
        self.specific    = specific
        self.averaged    = averaged

        # Check the dimension of the data
        if len(data.shape) > 2:
            raise Exception('The dimension of the data matrix is not (npts, ndim)!')
        if case == 1 and len(data.shape) == 1:
            data = data[:,np.newaxis]
        npts, ndimdata = data.shape
        if case != ndimdata and not conditioned:
            raise Exception('The dimension of the variables is %d, not %d!' % (ndimdata, case))
        elif case >= ndimdata and conditioned:
            raise Exception('The dimension of the variables should be larger than %d, not %d!' % (ndimdata, case))
        self.npts = npts
        self.case = case
        self.data = data

        # Check xyindex
        ndim = data.shape[1]
        self.__check_xyindex(xyindex, ndim)

        # Initiate the PDF computer
        if approach in kde_approaches:
            self.computer = pdf_computer(approach=approach, bandwidth=bandwidth, kernel=kernel)
            # 1D
            if self.case == 1 and not conditioned:
                self.__computeInfo1D_kde()
            elif self.case == 1 and conditioned:
                self.__computeInfo1D_conditioned_kde()

            # 2D
            if self.case == 2 and not conditioned:
                self.__computeInfo2D_kde()
            elif self.case == 2 and conditioned:
                self.__computeInfo2D_conditioned_kde()

            # 3D
            if self.case == 3 and not conditioned:
                self.__computeInfo3D_kde()
            elif self.case == 3 and conditioned:
                self.__computeInfo3D_conditioned_kde()

        elif approach in knn_approaches:
            knn = knn_cuda if approach is 'knn_cuda' else knn_scipy
            self.k = k
            # 1D
            if self.case == 1 and not conditioned:
                self.__computeInfo1D_knn()
            elif self.case == 1 and conditioned:
                self.__computeInfo1D_conditioned_knn()

            # 2D
            if self.case == 2 and not conditioned:
                self.__computeInfo2D_knn()
            elif self.case == 2 and conditioned:
                self.__computeInfo2D_conditioned_knn()

            # 3D
            if self.case == 3 and not conditioned:
                self.__computeInfo3D_knn()
            elif self.case == 3 and conditioned:
                self.__computeInfo3D_conditioned_knn()

        # Assemble all the information values into a Pandas series format
        self.__assemble()

    def __check_xyindex(self, xyindex, ndim):
        '''Check the xyz indexing.'''
        # 1D
        if self.case == 1:
            if xyindex is None:
                if conditioned: self.xlastind = 1
            elif isinstance(xyindex,list):
                if len(xyindex) == 1 and xyindex[0] <= ndim:
                    self.xlastind = xyindex[0]
                else:
                    raise Exception('xyindex is not correct for 1D case: ' + str(xyindex))
            else:
                raise Exception('Unknown type of xyindex %s' % str(type(xyindex)))
        # 2D
        elif self.case == 2:
            if xyindex is None:
                if not conditioned: self.xlastind = 1
                elif conditioned:   self.xlastind, self.ylastind = 1, 2
            elif isinstance(xyindex,list):
                if len(xyindex) == 2 and xyindex[0] < xyindex[1] and xyindex[1] <= ndim:
                    self.xlastind, self.ylastind = xyindex[0], xyindex[1]
                else:
                    raise Exception('xyindex is not correct for 2D case: ' + str(xyindex))
            else:
                raise Exception('Unknown type of xyindex %s' % str(type(xyindex)))
        # 3D
        elif self.case == 3:
            if xyindex is None:
                if not conditioned: self.xlastind, self.ylastind = 1, 2
                elif conditioned:   self.xlastind, self.ylastind, self.zlastind = 1, 2, 3
            elif isinstance(xyindex,list):
                if len(xyindex) == 3 and xyindex[0] < xyindex[1] and xyindex[1] < xyindex[2] and xyindex[2] <= ndim:
                    self.xlastind, self.ylastind, self.zlastind = xyindex[0], xyindex[1], xyindex[2]
                else:
                    raise Exception('xyindex is not correct for 3D case: ' + str(xyindex))
            else:
                raise Exception('Unknown type of xyindex %s' % str(type(xyindex)))

    def __computeInfo1D_kde(self):
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
        _, pdfs   = computer.computePDF(data)

        # Compute information metrics
        self.hx = computeEntropy(pdfs, base=base, averaged=averaged)

    def __computeInfo1D_knn(self):
        '''
        Compute H(X) using KNN method.
        Input:
        Output: NoneType
        '''
        base       = self.base
        data       = self.data
        k          = self.k
        npts, ndim = data.shape

        # Compute the ball radius of the k nearest neighbor for each data point
        dist, _ = knn(querypts=data, refpts=data, k=k)
        radiusset = dist[:, -1]

        # Note that the number of nearest neighbors with ball radius radiusset is always k in the joint dataset
        kset = k*np.ones(npts)

        # Compute information metrics
        self.hx = computeEntropyKNN(npts, ndim, kset, radiusset, base)

    def __computeInfo1D_conditioned_kde(self):
        '''
        Compute H(X|W)
        '''
        base     = self.base
        data     = self.data
        computer = self.computer
        averaged = self.averaged
        npts, ndim = data.shape

        xlastind = self.xlastind

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs = computer.computePDF(data[:,range(0,xlastind)])
        _, wpdfs = computer.computePDF(data[:,range(xlastind,ndim)])

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs, base=base, averaged=averaged)    # H(W)
        self.hx    = computeEntropy(xpdfs, base=base, averaged=averaged)    # H(X)
        self.hxw   = computeEntropy(pdfs, base=base, averaged=averaged)     # H(X,W)
        self.hx_w  = self.hxw - self.hw                                     # H(X|W)

    def __computeInfo1D_conditioned_knn(self):
        '''
        Compute H(X|W) using KNN method
        '''
        base     = self.base
        data     = self.data
        computer = self.computer
        averaged = self.averaged
        k         = self.k
        npts, ndim = data.shape

        xlastind = self.xlastind

        # The dimensions for X and W
        xndim, wndim = xlastind, ndim-xlastind

        # Get the conditioned data set
        xdata = data[:,range(0,xlastind)]
        wdata = data[:,range(xlastind,ndim)]

        # Compute the ball radius of the k nearest neighbor for each data point
        dist, _ = knn(querypts=data, refpts=data, k=k)
        rset = dist[:, -1][:, np.newaxis]

        # Get the number of nearest neighbors for X and W based on the ball radius
        distw, _ = knn(querypts=wdata, refpts=wdata, k=npts)
        distx, _ = knn(querypts=xdata, refpts=xdata, k=npts)
        kwset    = np.sum(distw < rset, axis=1)
        kxset    = np.sum(distx < rset, axis=1)

        # Note that the number of nearest neighbors with ball radius rset for XW is always k in the joint dataset
        kset = k*np.ones(npts)

        # Compute information metrics
        self.hxw  = computeEntropyKNN(npts, ndim, kset, rset, base)
        self.hw   = computeEntropyKNN(npts, wndim, kwset, rset, base)
        self.hx   = computeEntropyKNN(npts, xndim, kxset, rset, base)
        self.hx_w = self.hxw - self.hw

   def __computeInfo2D_kde(self):
        '''
        Compute H(X), H(Y), H(X|Y), H(Y|X), I(X;Y) using KNN method
        '''
        base     = self.base
        data     = self.data
        computer = self.computer
        averaged = self.averaged
        k        = self.k
        npts, ndim = data.shape

        xlastind = self.xlastind

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs = computer.computePDF(data[:,range(0,xlastind)])
        _, ypdfs = computer.computePDF(data[:,range(xlastind,ndim)])

        # Compute H(X), H(Y) and H(X,Y)
        # print xpdfs
        self.hx  = computeEntropy(xpdfs, base=base, averaged=averaged)  # H(X)
        self.hy  = computeEntropy(ypdfs, base=base, averaged=averaged)  # H(Y)
        self.hxy = computeEntropy(pdfs, base=base, averaged=averaged)   # H(X,Y)
        self.hy_x = self.hxy - self.hx                                  # H(Y|X)
        self.hx_y = self.hxy - self.hy                                  # H(X|Y)
        self.ixy  = self.hx + self.hy - self.hxy                        # I(X;Y)

   def __computeInfo2D_knn(self):
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
        npts, ndim = data.shape

        xlastind = self.xlastind

        # The dimensions for X and Y
        xndim, yndim = xlastind, ndim-xlastind

        # Get the conditioned data set
        xdata = data[:,range(0,xlastind)]
        ydata = data[:,range(xlastind,ndim)]

        # Compute the ball radius of the k nearest neighbor for each data point
        dist, _ = knn(querypts=data, refpts=data, k=k)
        rset    = dist[:, -1][:, np.newaxis]

        # Get the number of nearest neighbors for X and Y based on the ball radius
        disty, _ = knn(querypts=ydata, refpts=ydata, k=npts)
        distx, _ = knn(querypts=xdata, refpts=xdata, k=npts)
        kyset    = np.sum(disty < rset, axis=1)
        kxset    = np.sum(distx < rset, axis=1)

        # Note that the number of nearest neighbors with ball radius rset for XW is always k in the joint dataset
        kset = k*np.ones(npts)

        # Compute information metrics
        self.hxy  = computeEntropyKNN(npts, ndim, kset, rset, base)
        self.hy   = computeEntropyKNN(npts, yndim, kyset, rset, base)
        self.hx   = computeEntropyKNN(npts, xndim, kxset, rset, base)
        self.hx_y = self.hxy - self.hy
        self.hy_x = self.hxy - self.hx
        self.ixy  = self.hx + self.hy - self.hxy                        # I(X;Y)

    def __computeInfo2D_conditioned_kde(self):
        '''
        Compute H(X|W), H(Y|W), H(X,Y|W), I(X,Y|W)
        '''
        base       = self.base
        data       = self.data
        computer   = self.computer
        averaged   = self.averaged
        npts, ndim = data.shape

        xlastind, ylastind = self.xlastind, self.ylastind

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
        _, ypdfs  = computer.computePDF(data[:,range(xlastind,ylastind)])
        _, wpdfs  = computer.computePDF(data[:,range(ylastind,ndim)])
        _, xypdfs = computer.computePDF(data[:,range(0,ylastind)])
        _, xwpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,ndim)])
        _, ywpdfs = computer.computePDF(data[:,range(xlastind,ndim)])

        # Compute all the entropies
        self.hw    = computeEntropy(wpdfs, base=base, averaged=averaged)    # h(w)
        self.hx    = computeEntropy(xpdfs, base=base, averaged=averaged)    # h(x)
        self.hy    = computeEntropy(ypdfs, base=base, averaged=averaged)    # h(y)
        self.hxy   = computeEntropy(xypdfs, base=base, averaged=averaged)   # h(x,y)
        self.hxw   = computeEntropy(xwpdfs, base=base, averaged=averaged)   # h(x,w)
        self.hyw   = computeEntropy(ywpdfs, base=base, averaged=averaged)   # h(y,w)
        self.hxyw  = computeEntropy(pdfs, base=base, averaged=averaged)     # h(x,y,w)
        self.hx_w  = self.hxw - self.hw                                     # h(x|w)
        self.hy_w  = self.hyw - self.hw                                     # h(y|w)
        self.hx_y  = self.hxy - self.hy                                     # h(x|y)
        self.hy_x  = self.hxy - self.hx                                     # h(y|x)

        # Compute all the conditional mutual information
        self.ixy   = self.hx + self.hy - self.hxy                           # I(X;Y)
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw              # I(X;Y|W)

    def __computeInfo2D_conditioned_knn(self):
        '''
        Compute H(X|W), H(Y|W), H(X,Y|W), I(X,Y|W) using KNN method
        '''
        base       = self.base
        data       = self.data
        computer   = self.computer
        averaged   = self.averaged
        k          = self.k
        npts, ndim = data.shape

        xlastind, ylastind = self.xlastind, self.ylastind

        # The dimensions for X, Y and W
        xndim, yndim, wndim= xlastind, ylastind-xlastind, ndim-ylastind

        # Get the conditioned data set
        wdata  = data[:,range(xlastind,ndim)]
        xwdata = data[:,range(0,xlastind)+range(ylastind,ndim)]
        ywdata = data[:,range(xlastind,ndim)]

        # Compute the ball radius of the k nearest neighbor for each data point
        dist, _ = knn(querypts=data, refpts=data, k=k)
        rset    = dist[:, -1][:, np.newaxis]

        # Get the number of nearest neighbors for X and Y based on the ball radius
        distyw, _ = knn(querypts=ywdata, refpts=ywdata, k=npts)
        distxw, _ = knn(querypts=xwdata, refpts=xwdata, k=npts)
        distw, _  = knn(querypts=wdata, refpts=wdata, k=npts)
        kywset    = np.sum(distyw < rset, axis=1)
        kxwset    = np.sum(distxw < rset, axis=1)
        kwset     = np.sum(distw < rset, axis=1)

        # Note that the number of nearest neighbors with ball radius rset for XW is always k in the joint dataset
        kset = k*np.ones(npts)

        # Compute information metrics
        self.hw   = computeEntropyKNN(npts, wndim, kwset, rset, base)
        self.hxw  = computeEntropyKNN(npts, xndim+wndim, kxwset, rset, base)
        self.hyw  = computeEntropyKNN(npts, yndim+wndim, kywset, rset, base)
        self.hxyw = computeEntropyKNN(npts, ndim, kset, rset, base)
        self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw              # I(X;Y|W)

    def __computeInfo3D_kde(self):
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

        xlastind, ylastind = self.xlastind, self.ylastind

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
        _, ypdfs  = computer.computePDF(data[:,range(xlastind,ylastind)])
        _, zpdfs  = computer.computePDF(data[:,range(ylastind,ndim)])
        _, xypdfs = computer.computePDF(data[:,range(0,ylastind)])
        _, xzpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,ndim)])
        _, yzpdfs = computer.computePDF(data[:,range(xlastind,ndim)])
        # _, xpdfs  = computer.computePDF(data[:,[0]])
        # _, ypdfs  = computer.computePDF(data[:,[1]])
        # _, zpdfs  = computer.computePDF(data[:,[2]])
        # _, xypdfs = computer.computePDF(data[:,[0,1]])
        # _, xzpdfs = computer.computePDF(data[:,[0,2]])
        # _, yzpdfs = computer.computePDF(data[:,[1,2]])

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

    def __computeInfo3D_conditioned_kde(self):
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

        xlastind, ylastind, zlastind = self.xlastind, self.ylastind, self.zlastind

        # Compute the pdfs
        _, pdfs  = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
        _, ypdfs  = computer.computePDF(data[:,range(xlastind,ylastind)])
        _, zpdfs  = computer.computePDF(data[:,range(ylastind,zlastind)])
        _, wpdfs  = computer.computePDF(data[:,range(zlastind,ndim)])
        _, xypdfs = computer.computePDF(data[:,range(0,ylastind)])
        _, xzpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,zlastind)])
        _, yzpdfs = computer.computePDF(data[:,range(xlastind,zlastind)])
        _, xwpdfs  = computer.computePDF(data[:,range(0,xlastind)+range(zlastind,ndim)])
        _, ywpdfs  = computer.computePDF(data[:,range(xlastind,ylastind)+range(zlastind,ndim)])
        _, zwpdfs  = computer.computePDF(data[:,range(ylastind,ndim)])
        _, xywpdfs = computer.computePDF(data[:,range(0,ylastind)+range(zlastind,ndim)])
        _, yzwpdfs = computer.computePDF(data[:,range(ylastind,ndim)])
        _, xzwpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,ndim)])
        # _, xpdfs   = computer.computePDF(data[:,[0]])
        # _, ypdfs   = computer.computePDF(data[:,[1]])
        # _, zpdfs   = computer.computePDF(data[:,[2]])
        # _, wpdfs   = computer.computePDF(data[:,3:])
        # _, xypdfs  = computer.computePDF(data[:,[0,1]])
        # _, xzpdfs  = computer.computePDF(data[:,[0,2]])
        # _, yzpdfs  = computer.computePDF(data[:,[1,2]])
        # _, xwpdfs  = computer.computePDF(data[:,[0]+range(3,ndim)])
        # _, ywpdfs  = computer.computePDF(data[:,[1]+range(3,ndim)])
        # _, zwpdfs  = computer.computePDF(data[:,[2]+range(3,ndim)])
        # _, xywpdfs = computer.computePDF(data[:,[0,1]+range(3,ndim)])
        # _, yzwpdfs = computer.computePDF(data[:,[1,2]+range(3,ndim)])
        # _, xzwpdfs = computer.computePDF(data[:,[0,2]+range(3,ndim)])

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

    # def __computeInfo2D_multivariate_conditioned(self):
    #     '''
    #     Compute H(Xset|W), H(Yset|W), H(Xset,Yset|W), I(Xset,Yset|W)
    #     used for computing the accumulated information transfer (AIT)
    #     '''
    #     base       = self.base
    #     data       = self.data
    #     computer   = self.computer
    #     averaged   = self.averaged
    #     npts, ndim = data.shape
    #     pdfs     = self.pdfs

    #     xlastind, ylastind = self.xlastind, self.ylastind

    #     # Compute the pdfs
    #     _, xpdfs  = computer.computepdf(data[:,range(0,xlastind)])
    #     _, ypdfs  = computer.computepdf(data[:,range(xlastind,ylastind)])
    #     _, wpdfs  = computer.computepdf(data[:,range(ylastind,ndim)])
    #     _, xypdfs = computer.computepdf(data[:,range(0,ylastind)])
    #     _, xwpdfs = computer.computepdf(data[:,range(0,xlastind)+range(ylastind,ndim)])
    #     _, ywpdfs = computer.computepdf(data[:,range(xlastind,ndim)])

    #     # compute all the entropies
    #     self.hw    = computeEntropy(wpdfs, base=base, averaged=averaged)    # h(w)
    #     self.hx    = computeEntropy(xpdfs, base=base, averaged=averaged)    # h(x)
    #     self.hy    = computeEntropy(ypdfs, base=base, averaged=averaged)    # h(y)
    #     self.hxy   = computeEntropy(xypdfs, base=base, averaged=averaged)   # h(x,y)
    #     self.hxw   = computeEntropy(xwpdfs, base=base, averaged=averaged)   # h(x,w)
    #     self.hyw   = computeEntropy(ywpdfs, base=base, averaged=averaged)   # h(y,w)
    #     self.hxyw  = computeEntropy(pdfs, base=base, averaged=averaged)     # h(x,y,w)
    #     self.hx_w  = self.hxw - self.hw                                     # h(x|w)
    #     self.hy_w  = self.hyw - self.hw                                     # h(y|w)
    #     self.hx_y  = self.hxy - self.hy                                     # h(x|y)
    #     self.hy_x  = self.hxy - self.hx                                     # h(y|x)

    #     # Compute all the conditional mutual information
    #     self.ixy   = self.hx + self.hy - self.hxy                           # I(X;Y)
    #     self.ixy_w = self.hxw + self.hyw - self.hw - self.hxyw              # I(X;Y|W)

    # def computeInfo2D_multiple_conditioned(self, xlastind, ylastind):
    #     '''
    #     Compute H(Xset|W), H(Yset|W), H(Xset,Yset|W), I(Xset,Yset|W)
    #     used for computing the accumulated information transfer (AIT)
    #     '''
    #     base       = self.base
    #     data       = self.data
    #     computer   = self.computer
    #     averaged   = self.averaged
    #     npts, ndim = data.shape

    #     if xlastind > ylastind:
    #         raise Exception("xlastind %d is larger than ylastind %d" % (xlastind, ylastind))
    #     if xlastind > ndim-1:
    #         raise Exception("xlastind %d is larger than the maximum dimension" % xlastind)
    #     if ylastind > ndim-1:
    #         raise Exception("ylastind %d is larger than the maximum dimension" % ylastind)

    #     # Compute the pdfs
    #     _, pdfs   = computer.computePDF(data)
    #     # _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
    #     # _, ypdfs  = computer.computePDF(data[:,range(xlastind,ylastind)])
    #     _, wpdfs  = computer.computePDF(data[:,range(ylastind,ndim)])
    #     # _, xypdfs = computer.computePDF(data[:,range(0,ylastind)])
    #     _, xwpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,ndim)])
    #     _, ywpdfs = computer.computePDF(data[:,range(xlastind,ndim)])

    #     # Compute all the conditional mutual information
    #     # Calculate the log of pdf
    #     pdfs_log, wpdfs_log    = np.ma.log(pdfs), np.ma.log(wpdfs)
    #     xwpdfs_log, ywpdfs_log = np.ma.log(xwpdfs), np.ma.log(ywpdfs)
    #     pdfs_log = (pdfs_log.filled(0) + wpdfs_log.filled(0) - xwpdfs_log.filled(0) - ywpdfs_log.filled(0)) / np.log(base)

    #     # Normalize the joint PDF
    #     pdfs_norm = pdfs / pdfs.sum()

    #     # The conditional probability
    #     return np.sum(pdfs_norm*pdfs_log)

    def __assemble(self):
        '''
        Assemble all the information values into a Pandas series format
        Output: NoneType
        '''
        if self.case == 1 and not self.conditioned:
            self.allInfo = pd.Series(self.hx, index=['H(X)'], name='ordinary')

        elif self.case == 1 and self.conditioned:
            self.allInfo = pd.Series([self.hx, self.hx_w], index=['H(X)', 'H(X|W)'], name='ordinary')

        elif self.case == 2 and not self.conditioned:
            self.allInfo = pd.Series([self.hx, self.hy, self.hx_y, self.hy_x, self.ixy],
                                     index=['H(X)', 'H(Y)', 'H(X|Y)', 'H(Y|X)', 'I(X;Y)'],
                                     name='ordinary')

        elif self.case == 2 and self.conditioned:
            self.allInfo = pd.Series([self.hx, self.hx_y, self.hxyw-self.hyw, self.ixy_w],
                                     index=['H(X)', 'H(X|Y)', 'H(X|Y,W)', 'I(X;Y|W)'],
                                     name='ordinary')

        elif self.case == 3 and not self.conditioned:
            self.allInfo = pd.Series([self.ixz, self.iyz, self.itot, self.ii, self.r, self.s, self.uxz, self.uyz, self.rmin, self.isource, self.rmmi],
                                     index=['I(X;Z)', 'I(Y;Z)', 'I(X,Y;Z)', 'II', 'R(Z;Y,X)', 'S(Z;Y,X)', 'U(Z,X)', 'U(Z,Y)', 'Rmin', 'Isource', 'RMMI'],
                                     name='ordinary')

        elif self.case == 3 and self.conditioned:
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
        if self.case == 1 and not self.conditioned:
            nx           = npts
            scalingbase  = np.log(nx) / np.log(base)
            self.hx_norm = self.hx / scalingbase
            # assemble it to pandas series
            norm_df = pd.Series(self.hx_norm, index=['H(X)'], name='norm')

        # 1D - conditioned
        # conditioned (percentage): H(X)
        # conditioned (magnitude):  Hmax(X,W) = log(Nx*Nw)/log(base)
        if self.case == 1 and self.conditioned:
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
        if self.case == 2 and not self.conditioned:
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
        if self.case == 2 and self.conditioned:
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
        if self.case == 3 and not self.conditioned:
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
        if self.case == 3 and self.conditioned:
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
    # normalize pdf if not averaged
    if not averaged:
        pdfs = pdfs / np.sum(pdfs)

    # Calculate the log of pdf
    pdfs_log = np.ma.log(pdfs)
    pdfs_log = pdfs_log.filled(0) / np.log(base)

    # Calculate H(X)
    if averaged:
        return -np.mean(pdfs_log)
    elif not averaged:
        return -np.sum(pdfs*pdfs_log)

def computeEntropyKNN(npts, ndim, kset, radiusset, base=2):
    '''
    Compute the entropy based on the k-nearest-neighbor method.
    Inputs:
    npts      -- the number of datapoints [int]
    ndim      -- the number of dimension [int]
    kset      -- the number of nearest neighbors for each data points within radiusset [a numpy array with shape (npts,)]
    radiusset -- the ball radius for each data points [a numpy array with shape (npts,)]
    base      -- the logrithmatic base (the default is 2) [float/int]
    Output:
    entropy [float]
    '''
    from scipy.special import digamma

    # Compute the volumn of ndim dimension (maximum norm)
    vdx = 2.**ndim

    # Compute the radius term
    # pdfs_log  = np.ma.log(pdfs)
    # pdfs_log  = pdfs_log.filled(0) / np.log(base)
    rd = np.mean(np.log(radiusset) / np.log(base))*ndim

    # Compute the k-digamma term
    kd = np.mean(digamma(kset))

    # Compute the entropy
    return digamma(npts) - kd + rd + vdx

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


def computeMI(data, approach='kde_c', bandwidth='silverman', kernel='gaussian', base=2, xyindex=None):
    '''
    Compute the mutual information I(X;Y) based on the original formula (not the average version).
    Input:
    data        -- the data [numpy array with shape (npoints, ndim)]
    approach    -- the code for computing PDF by using KDE
    kernel      -- the kernel type [string]
    bandwith    -- the band with of the kernel [string or float]
    base        -- the logrithmatic base (the default is 2) [float/int]
    xyindex     -- a list of index indicating the position of the involved variable set, used for computeInfo*D_multivariate*
                    1D: [xlastind], 2D: [xlastind, ylastind], 3D: [xlastind,ylastind,zlastind]
                    note that xlastind < ylastind < zlastind <= len(pdfs.shape)
                    if None, used for computeInfo*D*
    '''
    # Check the dimension of the data
    if len(data.shape) > 2:
        raise Exception('The dimension of the data matrix is not (npts, ndim)!')

    npts, ndim = data.shape

    # Initiate the PDF computer
    computer = pdf_computer(approach=approach, bandwidth=bandwidth, kernel=kernel)

    # Compute the pdfs
    if xyindex:
        xlastind = xyindex[0]
        _, pdfs  = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
        _, ypdfs  = computer.computePDF(data[:,range(xlastind,ndim)])
    else:
        _, pdfs  = computer.computePDF(data)
        _, xpdfs = computer.computePDF(data[:,[0]])
        _, ypdfs = computer.computePDF(data[:,[1]])

    # Normalize PDF
    pdfsn = pdfs / np.sum(pdfs)

    # Calculate the log of pdf
    pdfs_log  = np.ma.log(pdfs)
    pdfs_log  = pdfs_log.filled(0) / np.log(base)
    xpdfs_log = np.ma.log(xpdfs)
    xpdfs_log = xpdfs_log.filled(0) / np.log(base)
    ypdfs_log = np.ma.log(ypdfs)
    ypdfs_log = ypdfs_log.filled(0) / np.log(base)

    return np.sum((pdfs_log - xpdfs_log - ypdfs_log)*pdfsn)


def computeCMI(data, approach='kde_c', bandwidth='silverman', kernel='gaussian', base=2, xyindex=None):
    '''
    Compute the conditional mutual information I(X;Y|Z) based on the original formula (not the average version).
    Input:
    data        -- the data [numpy array with shape (npoints, ndim)]
    approach    -- the code for computing PDF by using KDE
    kernel      -- the kernel type [string]
    bandwith    -- the band with of the kernel [string or float]
    base        -- the logrithmatic base (the default is 2) [float/int]
    xyindex     -- a list of index indicating the position of the involved variable set, used for computeInfo*D_multivariate*
                    1D: [xlastind], 2D: [xlastind, ylastind], 3D: [xlastind,ylastind,zlastind]
                    note that xlastind < ylastind < zlastind <= len(pdfs.shape)
                    if None, used for computeInfo*D*
    '''
    # Check the dimension of the data
    if len(data.shape) > 2:
        raise Exception('The dimension of the data matrix is not (npts, ndim)!')

    npts, ndim = data.shape

    # Initiate the PDF computer
    computer = pdf_computer(approach=approach, bandwidth=bandwidth, kernel=kernel)

    # Compute the pdfs
    if xyindex:
        xlastind, ylastind = xyindex[0], xyindex[1]
        _, pdfs  = computer.computePDF(data)
        _, xpdfs  = computer.computePDF(data[:,range(0,xlastind)])
        _, ypdfs  = computer.computePDF(data[:,range(xlastind,ylastind)])
        _, wpdfs  = computer.computePDF(data[:,range(ylastind,ndim)])
        _, xypdfs = computer.computePDF(data[:,range(0,ylastind)])
        # _, xwpdfs = computer.computePDF(data[:,range(0,xlastind)+range(ylastind,ndim)])
        # _, ywpdfs = computer.computePDF(data[:,range(xlastind,ndim)])
    else:
        _, pdfs  = computer.computePDF(data)
        _, xpdfs = computer.computePDF(data[:,[0]])
        _, ypdfs = computer.computePDF(data[:,[1]])
        _, wpdfs  = computer.computePDF(data[:,2:])
        _, xypdfs = computer.computePDF(data[:,[0,1]])
    xy_wpdfs  = xypdfs / wpdfs
    x_wpdfs   = xpdfs / wpdfs
    y_wpdfs   = ypdfs / wpdfs

    # Normalize PDF
    pdfsn = pdfs / np.sum(pdfs)

    # Calculate the log of pdf
    pdfs_log  = np.ma.log(xy_wpdfs)
    pdfs_log  = pdfs_log.filled(0) / np.log(base)
    xpdfs_log = np.ma.log(x_wpdfs)
    xpdfs_log = xpdfs_log.filled(0) / np.log(base)
    ypdfs_log = np.ma.log(y_wpdfs)
    ypdfs_log = ypdfs_log.filled(0) / np.log(base)

    return np.sum((pdfs_log - xpdfs_log - ypdfs_log)*pdfsn)
