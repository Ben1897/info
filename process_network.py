# A process network calculating the SUR relationship between a set of nodes
#
# @Author: Peishi Jiang <Ben1897>
# @Date:   2017-02-19T14:27:35-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Peishi
# @Last modified time: 2017-02-23T15:30:24-06:00

import numpy as np
import pandas as pd
from itertools import combinations

# Self-library
from info import info
from pdf_computer import pdfComputer
from sst import conductSST


class processNetwork(object):

    def __init__(self, data, nbins, maxlag=10):
        '''
        Inputs:
        data   -- the data [numpy array with shape (npoints, ndim)]
        nbins  -- the number of bins in all the dimensions
        maxlag --  the maximum time lag [int]
        '''
        self.npoints, self.nnodes = data.shape
        if self.nnodes <= 1:
            raise Exception('Less than two nodes!')
        self.data = data
        self.maxlag = maxlag

        # Number of bins in each dimension
        if len(nbins) != self.nnodes:
            raise Exception('The number of bins'' dimensions do not match with the number of the variables')
        self.nbins = nbins

        # Compute the mutual information between each pair and conduct the
        # statistical significance test on it
        self.__computeMI()

        # Compute the redundancy, unique and synergy information
        self.__computeSUR()

        # Reformat all the information (i.e., s, u, r and t) into the Pandas
        # DataFrame format
        self.__formattoPandas()

    def write_to_file(self):
        pass

    def __computeMI(self):
        '''
        Compute the mutual information between each pair and conduct the
        statistical significance test on it.
        The mutual information is calculated between each target node and other source node
        (including the target node itself and every node with the maximum time-delayed maxlag).
        Therefore, the shape of all the mutual information is (nnodes*(2*maxlag+1), nnodes).
        Output: NoneType
        '''
        # Create the maxtrix for storing mutual information, such that
        #                         n1(t)             ...          nn(t)
        # n1(t-maxlag)    I(n1(t),n1(t-maxlag))     ...   I(nn(t),n1(t-maxlag))
        # ...
        # n1(t+maxlag)    I(n1(t),n1(t+maxlag))     ...   I(nn(t),n1(t+maxlag))
        # ...
        # nn(t-maxlag)    I(n1(t),nn(t-maxlag))     ...   I(nn(t),nn(t-maxlag))
        # ...
        # nn(t+maxlag)    I(n1(t),nn(t+maxlag))     ...   I(nn(t),nn(t+maxlag))

        # Get the number of nodes, the maximum lag and the data
        nn, tau, data = self.nnodes, self.maxlag, self.data

        # Get the width of the lag set can create the lag set
        # The lag set is [-maxlag, -maxlag+1, ..., 0, ..., maxlag]
        ns, width, lagset = nn*(2*tau+1), 2*tau+1, np.arange(-tau, tau+1, 1, dtype=int)

        # Get the number of source nodes
        ns = nn*width

        # Create an empty matrix for storing all the mutual information
        mi = np.zeros([ns, nn], dtype=float)

        # Get the number of bins for all the dimensions
        nbins = self.nbins

        # Compute the mutual information for each pair
        # Note that i is from 1 to nn, however, j is from i to nn. This is
        # to avoid the duplicated calculation of mutual information because
        # I(ni(t), nj(t+lag)) = I(ni(t-lag), nj(t))
        for i in range(nn):
            for j in range(i, nn):
                for k1 in range(lagset.size):
                    # Get the lag
                    lag = lagset[k1]
                    if lag > 0:     # the source is delayed
                        dtarget = data[:-lag, i]
                        dsource = data[lag:, j]
                    elif lag < 0:   # the target is delayed
                        lag = -lag
                        dtarget = data[lag:, i]
                        dsource = data[:-lag, j]
                    elif lag == 0:  # no delay
                        dtarget = data[:, i]
                        dsource = data[:, j]

                    # Get the number of bins for dtarget and dsource
                    nx, ny = nbins[i], nbins[j]

                    # Calculate I(ni(t), nj(t+lag))
                    ind_i1, ind_i2 = width*j+k1, i
                    _, mi[ind_i1,ind_i2] = conductSST(dtarget, dsource, nx, ny)

                    # Assign I(ni(t), nj(t+lag)) to I(nj(t), ni(t-lag))
                    k2 = lagset.size - k1 - 1
                    ind_j1, ind_j2 = width*i+k2, j
                    mi[ind_j1,ind_j2] = mi[ind_i1,ind_i2]

        self.width  = width
        self.lagset = lagset
        self.ns     = ns
        self.mi     = mi

    def __computeSUR(self):
        '''
        Compute the synergistic, redundant and unique information for each combination
        of the nodes.
        Suppose for each target node i, the number of source node with significant
        mutual information with node i is nmi. The following information will be created:
            mistloc     -- the location of a source variable sharing a significant mutual
                           information with the target node i in self.mi[:, i], with # nmi
                           dtype: list with length nnodes
            mistcombloc -- all the combinations of the locations of source variables sharing a significant mutual
                           information with the target node i in self.mi[:, i] with # nmi*(nmi-1)/2
                           dtype: list with length nnodes
            cmi         -- the conditional mutual information of each pair
                           dtype: ndarray of object with shape (nnodes, nnodes*(2*maxlag+1), nnodes*(2*maxlag+1))
            r           -- the redundant information of each pair
                           dtype: ndarray of object with shape (nnodes, nnodes*(2*maxlag+1), nnodes*(2*maxlag+1))
            s           -- the synergistic information of each pair
                           dtype: ndarray of object with shape (nnodes, nnodes*(2*maxlag+1), nnodes*(2*maxlag+1))
            u           -- the unique information of each pair
                           dtype: ndarray of object with shape (nnodes, nnodes*(2*maxlag+1), nnodes*(2*maxlag+1))
            ii          -- the interaction information of each pair
                           dtype: ndarray of object with shape (nnodes, nnodes*(2*maxlag+1), nnodes*(2*maxlag+1))
        Outputs: NoneType
        '''
        nbins          = self.nbins
        nn, mi, lagset = self.nnodes, self.mi, self.lagset
        ns, width      = self.ns, self.width

        # Generate matrix for storing the location of a source variable
        # sharing a significant mutual information with the target node i in self.mi[:, i]
        funcmiloc = lambda mi: np.where(mi != 0)[0].tolist()
        mistloc   = map(funcmiloc, mi.T)

        # Generate all the combinations of the locations of source variables
        # sharing a significant mutual information with the target node i in self.mi[:, i]
        funcmicombloc = lambda mistloclist: list(combinations(mistloclist, 2))
        mistcombloc   = map(funcmicombloc, mistloc)

        # Create matrices for the redundant information, the unique information
        # the synergistic information and the transfer information
        r   = np.zeros([nn, ns, ns])
        s   = np.zeros([nn, ns, ns])
        u   = np.zeros([nn, ns, ns])
        cmi = np.zeros([nn, ns, ns])
        ii  = np.zeros([nn, ns, ns])

        # Generate the all the information
        for i in range(nn):
            combloc_i = mistcombloc[i]
            for combloc in combloc_i:
                loc1, loc2 = combloc[0], combloc[1]
                # Get the data of the source variable 1 with the lag
                j1   = loc1 / width
                lag1 = lagset[loc1 % width]
                # Get the data of the source variable 2 with the lag
                j2   = loc2 / width
                lag2 = lagset[loc2 % width]
                # Get the dimensions of the target and the two source variables
                nx, ny, nz = nbins[j2], nbins[j1], nbins[i]
                # Get the data
                dtar, ds1, ds2 = self.__getTargetSourceData(i, j1, j2, lag1, lag2)
                # Compute S, U, R and T
                pdfsolver  = pdfComputer(ndim=3, approach='kde', bandwidth='silverman',
                                         kernel='epanechnikov')
                _, pdf, _  = pdfsolver.computePDF(np.array([ds2, ds1, dtar]).T,
                                                  [nx, ny, nz])
                info_value = info(pdf)
                # Assign S, U, R and T to the matrices
                s[i, loc1, loc2] = info_value.s
                s[i, loc2, loc1] = info_value.s
                r[i, loc1, loc2] = info_value.r
                r[i, loc2, loc1] = info_value.r
                cmi[i, loc1, loc2] = info_value.iyz_x  # I(X1, Xtar|X2)
                cmi[i, loc2, loc1] = info_value.ixz_y  # I(X2, Xtar|X1)
                u[i, loc1, loc2] = info_value.r
                u[i, loc2, loc1] = info_value.r
                ii[i, loc1, loc2] = info_value.ii
                ii[i, loc2, loc1] = info_value.ii

        self.mistloc     = mistloc
        self.mistcombloc = mistcombloc
        self.r           = r
        self.s           = s
        self.u           = u
        self.cmi         = cmi
        self.ii          = ii

    def __formattoPandas(self):
        '''
        Format all the generated r, s, u and cmi into Pandas dataframe format.
        Outputs:
            pr, ps, pu, pcmi, pii -- the values of r, s, u, cmi and ii whose locations
                                   are in mistcombloc [pandas dataframe]
        '''
        mistcombloc, nn = self.mistcombloc, self.nnodes
        r, s, u, cmi, ii  = self.r, self.s, self.u, self.cmi, self.ii
        width, lagset   = self.width, self.lagset

        # Initialize the pandas dataframe for r, s, u and t
        keys = ['Target', 'Source 1', 'lag', 'Source 2', 'lag', 'Value']
        pr = pd.DataFrame(columns=keys)
        ps = pd.DataFrame(columns=keys)
        pu = pd.DataFrame(columns=keys)
        pcmi = pd.DataFrame(columns=keys)
        pii = pd.DataFrame(columns=keys)

        # Assign the values to pr, ps, pu, pt based on mistcombloc
        for i in range(nn):
            combloc_i = mistcombloc[i]

            for combloc in combloc_i:
                # Get the locations of the sources
                loc1, loc2 = combloc[0], combloc[1]

                # Get the index of the sources and the corresponding lag
                tar        = i + 1
                src1, src2 = (loc1 / width) + 1, (loc2 / width) + 1
                lag1, lag2 = lagset[loc1 % width], lagset[loc2 % width]

                # Redundancy
                pr.loc[len(pr)] = [tar, src1, lag1, src2, lag2, r[i, loc1, loc2]]
                pr.loc[len(pr)] = [tar, src2, lag2, src1, lag1, r[i, loc2, loc1]]

                # Synergy
                ps.loc[len(ps)] = [tar, src1, lag1, src2, lag2, s[i, loc1, loc2]]
                ps.loc[len(ps)] = [tar, src2, lag2, src1, lag1, s[i, loc2, loc1]]

                # Transfer
                pcmi.loc[len(pcmi)] = [tar, src1, lag1, src2, lag2, cmi[i, loc1, loc2]]
                pcmi.loc[len(pcmi)] = [tar, src2, lag2, src1, lag1, cmi[i, loc2, loc1]]

                # Unique info
                pu.loc[len(pu)] = [tar, src1, lag1, src2, lag2, u[i, loc1, loc2]]
                pu.loc[len(pu)] = [tar, src2, lag2, src1, lag1, u[i, loc2, loc1]]

                # Redundancy
                pii.loc[len(pii)] = [tar, src1, lag1, src2, lag2, ii[i, loc1, loc2]]
                pii.loc[len(pii)] = [tar, src2, lag2, src1, lag1, ii[i, loc2, loc1]]

        self.ps = ps
        self.pr = pr
        self.pcmi = pcmi
        self.pu = pu
        self.pii = pii

    def __getTargetSourceData(self, i, j1, j2, lag1, lag2):
        '''
        Get the data for the target variable and the two source variables with
        time lag (for 3D).
        Inputs:
            i          -- the index of the target variable [int]
            j1, j2     -- the indices of the two source variables [int]
            lag1, lag2 -- the time lags of the two source variables [int]
        Outputs:
            dtar       -- the array of the target variable [ndarray]
            ds1        -- the array of the source variable 1 [ndarray]
            ds2        -- the array of the source variable 2 [ndarray]
        '''
        data = self.data
        dtar, ds1, ds2 = data[:, i], data[:, j1], data[:, j2]

        # Adjust the three variables according to the source 1
        if lag1 > 0:   # the source 1 is delayed
            dtar = dtar[:-lag1]
            ds2 = ds2[:-lag1]
            ds1 = ds1[lag1:]
        elif lag1 < 0:  # the source 1 is forwarded
            lag1 = -lag1
            dtar = dtar[lag1:]
            ds2 = ds2[lag1:]
            ds1 = ds1[:-lag1]
        elif lag1 == 0:  # no delay
            dtar = dtar
            dtar = dtar
            ds1 = ds1

        # Adjust the three variables according to the source 2
        if lag2 > 0:   # the source 2 is delayed
            dtar = dtar[:-lag2]
            ds1 = ds1[:-lag2]
            ds2 = ds2[lag2:]
        elif lag2 < 0:  # the source 2 is forwarded
            lag2 = -lag2
            dtar = dtar[lag2:]
            ds1 = ds1[lag2:]
            ds2 = ds2[:-lag2]
        elif lag2 == 0:  # no delay
            dtar = dtar
            ds2 = ds2
            ds1 = ds1

        return dtar, ds1, ds2
