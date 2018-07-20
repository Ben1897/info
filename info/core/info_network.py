"""
The script is for computing the information transfer based on the data and the
causal network.
"""

import numpy as np

from copy import deepcopy

from .info import info, computeMI, computeCMI, computeMIKNN, computeCMIKNN
from ..utils.causal_network import causal_network
from ..utils.others import reorganize_data, dropna
from ..utils.sst import conditionalIndependence, independence, independenceSet, conditionalIndependenceSet

kde_approaches   = ['kde_c', 'kde_cuda', 'kde_cuda_general']
knn_approaches   = ['knn_cuda', 'knn_scipy', 'knn_sklearn', 'knn']
kde_approaches_p = ['kde_c']
knn_approaches_p = ['knn_scipy', 'knn_sklearn', 'knn']


class info_network(object):

    def __init__(self, data, causalDict, taumax=10,
                 approach='knn',
                 kernel='gaussian',  # parameters for KDE
                 k=5,                # parameters for KNN
                 base=np.e):
        """
        Input:
        data       -- the data [numpy array with shape (npoints, ndim)]
        causalDict -- dictionary of causal relationships, where
                      the keys are the variable at t time [int]
                      the values are the parents of the corresponding node [list of sets].
                      e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
        taumax     -- the maximum time lag for generating causal network [int]
        kernel     -- the kernel type [str]
        approach   -- the selected approach for computing PDF [string]
        base       -- the logrithmatic base [float]

        """
        self.data   = data
        self.taumax = int(taumax)
        self.base   = float(base)
        self.approach = approach
        self.kernel = kernel
        self.k      = k

        # Check whether the number of variables are larger than 1
        self.npoint, self.nvar = data.shape
        if self.nvar <= 1:
            raise Exception('Less than two nodes!')

        # Generate the causal network
        self.causalDict = causalDict
        self.originalCausalDict = causalDict
        self.network    = causal_network(causalDict, taumax)
        self.nnode      = self.network.nnodes
        if self.network.nvar != self.nvar:
            raise Exception('The numbers of variables in data and the causalDict are not equal!')


    def update_causalDict(self, causalDict, verbosity=1):
        """Update the causalDict."""
        # Update causalDict
        self.causalDict = causalDict
        taumax = self.taumax

        # Update the network
        self.network    = causal_network(causalDict, taumax)
        self.nnode      = self.network.nnodes

        # Update the lag functions (coupling strengths)
        self.generate_lagfunctions()

    def update_causalDict_thres(self, csthreshold, verbosity=1):
        """Update the causalDict based on the coupling strength threshold."""
        T, N          = self.data.shape
        causalDict    = deepcopy(self.causalDict)
        taumax        = self.taumax
        newcausalDict = dict.fromkeys(causalDict.keys())

        # Check whether the lagfuncs have been computed
        if not hasattr(self, 'lagfuncmit') or not hasattr(self, 'lagfuncmi'):
            self.generate_lagfunctions()

        # Generate the latest causalDict
        lags_causal = self.lagfuncmit > csthreshold
        for j in range(N):
            pa_old = causalDict[j]
            pa_new = []
            lags_causal_j = lags_causal[:,j,:]
            for i in range(N):
                pa_new_i = [(i,-l) for l, x in enumerate(lags_causal_j[i,:]) if x]
                pa_new += intersection(pa_old, pa_new_i)
            newcausalDict[j] = pa_new

        if verbosity:
            print ""
            print "The updated causalDict is:"
            print newcausalDict
            print ""

        # Update causalDict
        self.causalDict = newcausalDict

        # Update the network
        self.network    = causal_network(causalDict, taumax)
        self.nnode      = self.network.nnodes

        # Update the lag functions (coupling strengths)
        self.generate_lagfunctions()


    def update_causalDict_one(self, verbosity=1):
        """Update the causalDict so that the directed influence between two variables only include
           (1) one directed lagged edges (with the maximum coupling strength), and
           (2) (potentially) one contemporaneous directed edge."""
        T, N          = self.data.shape
        causalDict    = deepcopy(self.causalDict)
        taumax        = self.taumax
        newcausalDict = dict.fromkeys(causalDict.keys())

        # Check whether the lagfuncs have been computed
        if not hasattr(self, 'lagfuncmit') or not hasattr(self, 'lagfuncmi'):
            self.generate_lagfunctions()
        lagfunc = self.lagfuncmit

        # Generate the latest causalDict
        for j in range(N):
            pa_old = causalDict[j]
            pa_new = []
            for i in range(N):
                # the lag functions
                lagfunc_itoj = lagfunc[i,j,:]
                # check whether there is a contemporaneous directed link
                if (i,0) in pa_old:
                    pa_new += [(i,0)]
                # the lags of original parents of Xj from Xi
                pa_old_lag_i = [-l for ii, l in pa_old if ii == i and l != 0]
                if pa_old_lag_i == []:
                    continue
                # the lag functions in the original parents
                lagfunc_itoj_in = lagfunc_itoj[pa_old_lag_i]
                # get the lag with the maximum coupling strength
                lag_max = pa_old_lag_i[np.argmax(lagfunc_itoj_in)]
                # generate the new parents from Xi to Xj
                pa_new_i = [(i,-lag_max)]
                pa_new += pa_new_i
            newcausalDict[j] = pa_new

        if verbosity:
            print ""
            print "The updated causalDict is:"
            print newcausalDict
            print ""

        # Update causalDict
        self.causalDict = newcausalDict

        # Update the network
        self.network    = causal_network(newcausalDict, taumax)
        self.nnode      = self.network.nnodes

        # Update the lag functions (coupling strengths)
        self.generate_lagfunctions()


    def generate_lagfunctions(self):
        """Generate the lag functions as momentary information transfer and mutual information"""
        T, N       = self.data.shape
        base       = self.base
        data       = self.data
        approach   = self.approach
        kernel     = self.kernel
        k          = self.k
        causalDict = self.causalDict

        # Get the maximum time lag
        lags   = [pa[1] for paset in causalDict.values() for pa in paset]
        lagmax = np.absolute(lags).max()

        # Create an empty lagfunctions for MIT and MI
        self.lagfuncmit = np.zeros([N, N, lagmax+1])
        self.lagfuncmi  = np.zeros([N, N, lagmax+1])

        # Compute MIT and MI
        for i in range(N):
            for j in range(N):
                for l in range(lagmax+1):  # coupling strength X(i,-k) --> X(j,0)
                    # Do not compute the 'self' coupling strength
                    if i == j and l == 0: continue

                    snode, tnode = (i,-l), (j,0)

                    # Compute MI
                    # Get data for node1 and node2
                    data12 = reorganize_data(data, [snode, tnode])

                    # Drop the rows with nan values
                    data12n = dropna(data12)

                    # Calculate the mutual information of them
                    if approach in kde_approaches:
                        self.lagfuncmi[i,j,l] = computeMI(data=data12n, approach=approach, kernel=kernel, base=base)
                    elif approach in knn_approaches:
                        self.lagfuncmi[i,j,l] = computeMIKNN(data=data12n, k=k) / np.log(base)

                    # Compute MIT
                    # Get the condition for MIT
                    conditionset = self.network.search_mit_condition(snode, tnode)
                    # Get data for node1 and node2
                    data12cond = reorganize_data(data, [snode, tnode]+conditionset)

                    # Drop the rows with nan values
                    data12condn = dropna(data12cond)

                    # Calculate the conditional mutual information of them
                    if approach in kde_approaches:
                        self.lagfuncmit[i,j,l] = computeCMI(data=data12condn, approach=approach, kernel=kernel, base=base)
                    elif approach in knn_approaches:
                        self.lagfuncmit[i,j,l] = computeCMIKNN(data=data12condn, k=k) / np.log(base)


    def compute_total_infotrans(self, target_ind, sst=False, verbosity=1):
        """
        Compute the total information transfer to a target. I(Xtar; P(Xtar))

        Input:
        target_ind  -- the index of the target variable [int]

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k
        causalDict = self.causalDict
        nvar   = self.nvar

        target = (target_ind, 0)

        # Get the parents of the target
        pt = causalDict[target_ind]

        # Reorganize the data
        data1 = reorganize_data(data, [target]+pt)

        # Drop the nan values
        data1 = dropna(data1)
        if data1.shape[0] < 100:
            print 'Not enough time series datapoints (<100)!'
            return 0.
        inforesult = computeMIKNN(data1, k=k, xyindex=[1]) / np.log(base)

        if sst:
            sstresult = independenceSet(target, pt, data=data,
                                        approach=approach, k=k, kernel=kernel, base=base)
            return inforesult, sstresult

        return inforesult


    def compute_pairwise_infotrans(self, source_ind, target_ind, conditioned=True, sst=False, verbosity=1):
        """
        Compute the pairwise information transfer given the source and the target variables.
        I(Xtar;Xsrcs in P(Xtar) | the remaining P(Xtar))

        Input:
        source_ind  -- the index of the source variable [int]
        target_ind  -- the index of the target variable [int]
        conditioned -- whether including conditions [bool]

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k
        causalDict = self.causalDict
        nvar   = self.nvar

        target = (target_ind, 0)

        # # Check whether source_ind == target_ind
        # if source_ind == target_ind:
        #     raise Exception('The source variable is the same as the target variable!')

        # Get the parents of the target
        pt = causalDict[target_ind]

        # Get the parents of the target from the source variable
        pts = [p for p in pt if p[0] == source_ind]
        if not pts:
            if verbosity: print "Source variable %d does not influence the target %d" % (source_ind, target_ind)
            if sst:
                return 0., True
            else:
                return 0.

        # If not conditioned, just compute the mutual information
        if not conditioned:
            # Reorganize the data
            data1   = reorganize_data(data, [target]+pts)
            # Drop the nan values
            data1   = dropna(data1)
            if data1.shape[0] < 100:
                print 'Not enough time series datapoints (<100)!'
                return 0.
            inforesult = computeMIKNN(data1, k=k, xyindex=[1]) / np.log(base)

            if sst:
                sstresult = independenceSet(target, pts, data=data,
                                            approach=approach, k=k, kernel=kernel, base=base)
                return inforesult, sstresult

            return inforesult

        # If conditioned, compute the conditional mutual information which is I(target;pts | pt\pts)
        # Get the condition set
        w = list(set(pt) - set(pts))
        # Reorganize the data
        data2 = reorganize_data(data, [target] + pts + w)
        # Drop the nan values
        data2 = dropna(data2)
        # Compute the information transfer
        if w:
            inforesult = computeCMIKNN(data2, k=k, xyindex=[1, 1+len(pts)]) / np.log(base)
        else:
            inforesult = computeMIKNN(data2, k=k, xyindex=[1]) / np.log(base)

        # Conduct the significance test if required
        if sst:
            sstresult = conditionalIndependenceSet(target, pts, conditionset=w, data=data,
                                                   approach=approach, k=k, kernel=kernel, base=base)
            return inforesult, sstresult

        return inforesult


    def compute_2n_infotrans(self, source, target, conditioned=True, causalpath=True, sst=False, verbosity=1):
        """
        Compute the information transfer from a source node to a target node. (MITP or MIT)

        Input:
        source      -- the source node [set (var_index, lag)]
        target      -- the target node [set (var_index, lag)]
        conditioned -- whether including conditions [bool]
        causalpath  -- whether computing MITP or MIT [bool]

        Ouput:
        an instance from the class info based on the method __computeInfo2D_conditioned() or __computeInfo2D()

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k

        # If not conditioned, just compute the normal information metrics (not the momentary one)
        if not conditioned:
            # Reorganize the data
            data_required = reorganize_data(data, [target, source])
            # Drop nan
            data_required = dropna(data_required)
            # Compute the information transfer
            inforesult = info(case=2, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=False)
            # Conduct the significance test if required
            if sst:
                sstresult = independence(target, source, data=data, approach=approach, k=k, kernel=kernel, base=base)
                return inforesult, sstresult

            return inforesult

        # Check whether the two nodes are linked
        if network.check_links(source, target, verbosity=0) not in ['causalpath', 'directed']:
            print "The source %s and the target %s are not linked through a causal path or a contemporaneous link!" % (source, target)
            return None

        # Generate the MIT/MITP conditions
        if causalpath:
            w = network.search_mitp_condition(source, target, verbosity=verbosity)
        else:
            w = network.search_mit_condition(source, target, verbosity=verbosity)

        # Reorganize the data
        data_required = reorganize_data(data, [target, source] + w)

        # Drop nan
        data_required = dropna(data_required)

        # Compute the information transfer
        if w:
            inforesult = info(case=2, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=True)
        else:
            inforesult = info(case=2, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=False)

        # Conduct the significance test if required
        if sst:
            sstresult = conditionalIndependence(target, source, conditionset=w, data=data,
                                                approach=approach, k=k, kernel=kernel, base=base)
            return inforesult, sstresult

        return inforesult


    def compute_3n_infotrans(self, source1, source2, target, conditioned=True, verbosity=1):
        """
        Compute the information transfer from two source nodes to a target node. (MPID)

        Input:
        source1     -- the 1st source node [set (var_index, lag)]
        source2     -- the 2st source node [set (var_index, lag)]
        target      -- the target node [set (var_index, lag)]
        conditioned -- whether including conditions [bool]

        Output:
        an instance from the class info based on the method __computeInfo3D_conditioned() or __computeInfo3D()

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k

        # If not conditioned, just compute the normal information metrics (not the momentary one)
        if not conditioned:
            # Reorganize the data
            data_required = reorganize_data(data, [source1, source2, target])
            # Dropna
            data_required = dropna(data_required)
            # Compute the information transfer
            inforesult = info(case=3, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=False)

            return inforesult

        # Check whether each source node is linked with the target through a causal path
        if network.check_links(source1, target, verbosity=verbosity) not in ['causalpath', 'directed']:
            print "The source %s and the target %s are not linked through a causal path!" % (source1, target)
            return None
        if network.check_links(source2, target, verbosity=verbosity) not in ['causalpath', 'directed']:
            print "The source %s and the target %s are not linked through a causal path!" % (source2, target)
            return None

        # Generate the MIT/MITP conditions
        w = network.search_mpid_condition(source1, source2, target, verbosity=verbosity)

        # Reorganize the data
        data_required = reorganize_data(data, [source1, source2, target] + w)
        # Dropna
        data_required = dropna(data_required)
        # Compute the information transfer
        if w:
            inforesult = info(case=3, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=True)
        else:
            inforesult = info(case=3, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=False)

        return inforesult


    def compute_mitp(self, sources, target, conditioned=True, sst=False, verbosity=1):
        """Compute the momentary information transfer from sources to target.

        Input:
        sources     -- a list of source nodes [[set (var_index, lag)]]
        target      -- the target node [set (var_index, lag)]

        """
        network = self.network
        base    = self.base
        data    = self.data
        k       = self.k
        approach = self.approach
        kernel  = self.kernel

        # If not conditioned, just compute the normal information metrics (not the momentary one)
        if not conditioned:
            # Reorganize the data
            data_required = reorganize_data(data, [target] + sources)
            # Drop the nan
            data_required = dropna(data_required)
            # Compute the information transfer
            tit = computeMIKNN(data_required, k=k, xyindex=[1]) / np.log(base)        # distant causal history
            # Conduct the significance test if required
            if sst:
                ssttit = independenceSet(target, sources, data=data,
                                         approach=approach, k=k, kernel=kernel, base=base)
                return tit, ssttit

            return tit

        # Get the condition set and the parents of the target in the causal subgraph
        try:
            w, srcc, cpaths = network.search_cit_components(sources, target, mpid=True, verbosity=verbosity)
        except:
            print "Warning:!!!"
            print network.search_cit_components(sources, target)
            return None

        # Reorganize the data
        data1 = reorganize_data(data, [target]+srcc+w)
        # print data1.shape

        # Drop the nan values
        data1 = dropna(data1)
        xyindex1 = [1,1+len(srcc)]
        # print data21.shape, data22.shape
        if data1.shape[0] < 100:
            print 'Not enough time series datapoints (<100)!'
            return None

        # Compute the information from the immediate causal history and distant causal history
        mitp  = computeCMIKNN(data1, k=k, xyindex=xyindex1) / np.log(base) # immediate causal history

        # Conduct the significance test if required
        if sst:
            sstmitp = conditionalIndependenceSet(target, srcc, conditionset=w, data=data,
                                                 approach=approach, k=k, kernel=kernel, base=base)
            return mitp, sstmitp

        return mitp


    def compute_mpid(self, sources1, sources2, target, conditioned=True, verbosity=1):
        """Compute the momentary partial information decomposition from two sets of sources to the target.

        Input:
        sources1    -- a first list of source nodes [[set (var_index, lag)]]
        sources2    -- a second list of source nodes [[set (var_index, lag)]]
        target      -- the target node [set (var_index, lag)]

        """
        network = self.network
        base    = self.base
        data    = self.data
        k       = self.k
        approach = self.approach
        kernel  = self.kernel

        # If not conditioned, just compute the normal information metrics (not the momentary one)
        if not conditioned:
            # Reorganize the data
            data_required = reorganize_data(data, sources1 + sources2 + [target])
            # Drop the nan
            data_required = dropna(data_required)
            xyindex = [len(sources1), len(sources1)+len(sources2)]
            # Compute the information transfer
            inforesult = info(case=3, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=False, xyindex=xyindex)

            return inforesult

        # Generate the MPID conditions
        try:
            srcs1, srcs2, w = network.search_mpid_set_condition(sources1, sources2, target, verbosity=verbosity)
        except:
            print "Warning:!!!"
            print network.search_mpid_set_condition(sources1, sources2, target, verbosity=verbosity)
            return None

        # Reorganize the data
        # data_required = reorganize_data(data, sources1 + sources2 + [target] + w)
        data_required = reorganize_data(data, srcs1 + srcs2 + [target] + w)
        # Dropna
        data_required = dropna(data_required)
        # Compute the information transfer
        if w:
            xyindex = [len(srcs1), len(srcs1)+len(srcs2), len(srcs1)+len(srcs2)+1]
            inforesult = info(case=3, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=True, xyindex=xyindex)
        else:
            xyindex = [len(srcs1), len(srcs1)+len(srcs2)]
            inforesult = info(case=3, data=data_required, approach=approach, bandwidth='silverman',
                              kernel=kernel, k=k, base=base, conditioned=False, xyindex=xyindex)

        return inforesult


    def compute_cit(self, sources, target, sst=False, verbosity=1):
        """Compute the cumulative information transfer from sources to target.

        Input:
        sources     -- a list of source nodes [[set (var_index, lag)]]
        target      -- the target node [set (var_index, lag)]

        """
        network = self.network
        base    = self.base
        data    = self.data
        k       = self.k
        approach = self.approach
        kernel  = self.kernel

        # Get the parents of the target
        pt = network.search_parents(target)
        print "parents of the target:"
        print pt

        # Compute the total information
        # Reorganize the data
        data1 = reorganize_data(data, [target]+pt)
        data1 = dropna(data1)
        if data1.shape[0] < 100:
            print 'Not enough time series datapoints (<100)!'
            return citset, mitset, miset
        tit = computeMIKNN(data1, k=k, xyindex=[1]) / np.log(base)
        print data1

        # Get the condition set and the parents of the target in the causal subgraph
        try:
            w, ptc, cpaths = network.search_cit_components(sources, target, verbosity=verbosity)
        except:
            print "Warning:!!!"
            print network.search_cit_components(sources, target)
            return None

        # Reorganize the data
        data21   = reorganize_data(data, [target]+ptc+w)
        data22   = reorganize_data(data, [target]+w)

        # Drop the nan values
        data21   = dropna(data21)
        data22   = dropna(data22)
        # print data21.shape
        # datasize[i], dimsize[i] = data21.shape
        xyindex1 = [1,1+len(ptc)]
        # print data21.shape, data22.shape
        if data21.shape[0] < 100 or data22.shape[0] < 100:
            print 'Not enough time series datapoints (<100)!'
            return None

        # Compute the information from the immediate causal history and distant causal history
        cit  = computeCMIKNN(data21, k=k, xyindex=xyindex1) / np.log(base) # immediate causal history
        past = computeMIKNN(data22, k=k, xyindex=[1]) / np.log(base)        # distant causal history

        # Conduct the significance test if required
        if sst:
            ssttit = independenceSet(target, pt, data=data,
                                     approach=approach, k=k, kernel=kernel, base=base)
            sstcit = conditionalIndependenceSet(target, ptc, conditionset=w, data=data,
                                                approach=approach, k=k, kernel=kernel, base=base)
            sstpast = independenceSet(target, w, data=data,
                                      approach=approach, k=k, kernel=kernel, base=base)
            return cit, sstcit, past, sstpast, tit, ssttit

        return cit, past, tit


    def compute_mi_set(self, sst=False, verbosity=1):
        """
        Compute the pairwise mutual information.

        Input:
        sst -- whether conducting the significance test [bool]

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k
        nvar   = self.nvar

        inforesults = np.zeros([nvar, nvar])  # inforesults[:,0] is the total information transfer

        if sst:
            sstresults = np.zeros([nvar, nvar], dtype='bool')

        for i in range(nvar):
            varx = (i,0)
            for j in range(i+1,nvar):
                vary= (j,0)
                # Reorganize the data
                data_required = reorganize_data(data, [varx, vary])
                # Drop the nan values
                data_required = dropna(data_required)
                # Compute the total information
                if sst:
                    sstresults[i,j]  = independence(varx, vary, data=data,
                                                    approach=approach, k=k, kernel=kernel, base=base)
                    inforesults[i,j] = computeMIKNN(data_required, k=k) / np.log(base) # immediate causal history
                    sstresults[j,i] = sstresults[i,j]
                else:
                    inforesults[i,j] = computeMIKNN(data_required, k=k) / np.log(base) # immediate causal history

                inforesults[j,i] = inforesults[i,j]

        # Return
        if sst:
            return inforesults, sstresults
        else:
            return inforesults


    def compute_te_set(self, sst=False, verbosity=1):
        """
        Compute the pairwise transfer entropy.

        Input:
        sst -- whether conducting the significance test [bool]

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k
        nvar   = self.nvar

        inforesults = np.zeros([nvar, nvar])  # inforesults[:,0] is the total information transfer

        if sst:
            sstresults = np.zeros([nvar, nvar], dtype='bool')

        for src in range(nvar):
            srcnode = (src,-1)
            for tar in range(nvar):
                if src == tar: continue
                tarnode, w = (tar,0), (tar,-1)
                # Reorganize the data
                data_required = reorganize_data(data, [srcnode, tarnode, w])
                # Drop the nan values
                data_required = dropna(data_required)
                # Compute the total information
                if sst:
                    sstresults[src,tar]  = conditionalIndependence(tarnode, srcnode, conditionset=[w], data=data,
                                                               approach=approach, k=k, kernel=kernel, base=base)
                    inforesults[src,tar] = computeCMIKNN(data_required, k=k, xyindex=[1,2]) / np.log(base) # immediate causal history sstresults[tar,src] = sstresults[src,tar]
                else:
                    inforesults[src,tar] = computeCMIKNN(data_required, k=k, xyindex=[1,2]) / np.log(base) # immediate causal history

        # Return
        if sst:
            return inforesults, sstresults
        else:
            return inforesults


    def compute_pairwise_infotrans_set(self, conditioned=True, sst=False, verbosity=1):
        """
        Compute the pairwise information transfer.

        Input:
        conditioned -- whether including conditions [bool]

        """
        data    = self.data
        network = self.network
        base    = self.base
        approach = self.approach
        kernel = self.kernel
        k      = self.k
        causalDict = self.causalDict
        nvar   = self.nvar

        inforesults = np.zeros([nvar+1, nvar])  # inforesults[:,0] is the total information transfer

        if sst:
            sstresults = np.zeros([nvar+1, nvar], dtype='bool')

        for tar in range(nvar):
            # Compute the total information
            if sst:
                inforesults[-1,tar], sstresults[-1,tar] = self.compute_total_infotrans(target_ind=tar,
                                                                                     sst=sst,verbosity=verbosity)
            else:
                inforesults[-1,tar] = self.compute_total_infotrans(target_ind=tar,
                                                                  sst=sst,verbosity=verbosity)

            # Compute the pairwise information transfer
            for src in range(nvar):
                # if j == i: continue
                if sst:
                    # print self.compute_pairwise_infotrans(source_ind=source_ind,
                    #                                       target_ind=target_ind,
                    #                                       conditioned=conditioned,
                    #                                       sst=sst, verbosity=verbosity)
                    inforesults[src,tar], sstresults[src,tar] = self.compute_pairwise_infotrans(source_ind=src,
                                                                                                    target_ind=tar,
                                                                                                    conditioned=conditioned,
                                                                                                    sst=sst, verbosity=verbosity)
                else:
                    inforesults[src,tar] = self.compute_pairwise_infotrans(source_ind=src,
                                                                             target_ind=tar,
                                                                             conditioned=conditioned,
                                                                             sst=sst, verbosity=verbosity)

        # Return
        if sst:
            return inforesults, sstresults
        else:
            return inforesults


    def compute_2n_infotrans_set(self, source_ind, target_ind, conditioned=True, taumax=5, causalpath=True, verbosity=1):
        """
        Compute the information transfer from a source node to a target node with lags varying from 1 to taumax

        Input:
        source_ind -- the source variable index [ind]
        target_ind -- the target variable index [ind]
        conditioned -- whether including conditions [bool]
        taumax     -- the maximum lag between the source node and the target
        causalpath  -- whether computing MITP or MIT [bool]

        Ouput:

        """
        # Check whether taumax is out of the range of the allowed value self.taumax
        # If yes, set taumax to self.taumax
        if taumax > self.taumax:
            print 'The maximum lag %d is larger than the allowed value %d' % (taumax, self.taumax)
            print 'Reset taumax to %d' % self.taumax
            print ''
            taumax = self.taumax

        # Initialize the return array
        results = np.empty([taumax], dtype=object)

        # Loop
        for i in range(taumax):
            # Get the source and target node based on the lag tau
            tau = i + 1
            source, target = (source_ind, -tau), (target_ind, 0)

            # Compute the information transfer
            results[i] = self.compute_2n_infotrans(source, target, conditioned=conditioned,
                                                   causalpath=causalpath, verbosity=verbosity)

        # Return the results
        return results


    def compute_3n_infotrans_set(self, source1_ind, source2_ind, target_ind, conditioned=True, taumax=5, verbosity=1):
        """
        Compute the information transfer from two source nodes to a target node with lags varying from 1 to taumax

        Input:
        source1_ind -- the 1st source variable index [ind]
        source2_ind -- the 2nd source variable index [ind]
        target_ind -- the target variable index [ind]
        conditioned -- whether including conditions [bool]
        taumax     -- the maximum lag between the source node and the target

        Output:

        """
        # Check whether taumax is out of the range of the allowed value self.taumax
        # If yes, set taumax to self.taumax
        if taumax > self.taumax:
            print 'The maximum lag %d is larger than the allowed value %d' % (taumax, self.taumax)
            print 'Reset taumax to %d' % self.taumax
            print ''
            taumax = self.taumax

        # Initialize the return array
        results = np.empty([taumax, taumax], dtype=object)

        # Loop
        for i in range(taumax):
            for j in range(taumax):
                # Get the source and target node based on the lag tau
                tau1, tau2       = i + 1, j+1
                source1, source2 = (source1_ind, -tau1), (source2_ind, -tau2)
                target           = (target_ind, 0)

                # Compute the information transfer
                results[i, j] = self.compute_3n_infotrans(source1, source2, target,
                                                          conditioned=conditioned,
                                                          verbosity=verbosity)

        # Return the results
        return results


    def compute_mitp_set(self, taumax=None, conditioned=True, sst=False, verbosity=1):
        """Compute the momentary information transfer from sources to target.

        Input:
        sources     -- a list of source nodes [[set (var_index, lag)]]
        target      -- the target node [set (var_index, lag)]

        """
        network = self.network
        base    = self.base
        data    = self.data
        k       = self.k
        approach = self.approach
        kernel  = self.kernel
        nvar    = self.nvar

        if taumax is None:
            taumax = self.taumax

        # Initialization
        mitpset = np.zeros([nvar, nvar, taumax+1])

        if sst:
            sstmitpset = np.zeros([nvar, nvar, taumax+1], dtype='bool')

        # Compute the mitp
        for tar in range(nvar):
            target = (tar, 0)
            for src in range(nvar):
                for k in range(2, taumax+1):
                    sources = [(src, -l) for l in range(1, k)]
                    if sst:
                        # mitpset[tar,src,k], sstmitpset[tar,src,k] = self.compute_mitp(sources, target,
                        #                                                               conditioned=conditioned, sst=sst, verbosity=verbosity)
                        mitpset[src,tar,k], sstmitpset[src,tar,k] = self.compute_mitp(sources, target,
                                                                                      conditioned=conditioned, sst=sst, verbosity=verbosity)
                    else:
                        # mitpset[tar,src,k] = self.compute_mitp(sources, target,
                        #                                        conditioned=conditioned, sst=sst, verbosity=verbosity)
                        mitpset[src,tar,k] = self.compute_mitp(sources, target,
                                                               conditioned=conditioned, sst=sst, verbosity=verbosity)

        # Return
        if sst:
            return mitpset, sstmitpset
        else:
            return mitpset


    def compute_cit_set(self, sources, target, taumax, verbosity=1):
        """Compute the cumulative information transfer from source_ind_set to target_ind with increasing lags up to taumx.

        Input:
        sources     -- a list of source nodes [[set (var_index, lag)]]
        target      -- the target node [set (var_index, lag)]

        """
        data    = self.data
        base    = self.base
        network = self.network
        k       = self.k

        # Initialize
        citset   = np.zeros(taumax)
        pastset  = np.zeros(taumax)
        titset   = np.ones(taumax)
        datasize = np.zeros(taumax)
        dimsize  = np.zeros(taumax)

        # Get the parents of the target
        pt = network.search_parents(target)
        print "parents of the target:"
        print pt

        # Compute the total information
        # Reorganize the data
        data1 = reorganize_data(data, [target]+pt)
        data1 = dropna(data1)
        if data1.shape[0] < 100:
            print 'Not enough time series datapoints (<100)!'
            return citset, mitset, miset
        tit = computeMIKNN(data1, k=k, xyindex=[1]) / np.log(base)
        titset = tit*titset

        wrepeat = False

        # Compute I and P
        w1old, ptncold = [], []
        for i in range(taumax):
            print ""
            print ""
            print "Target and sources:"
            print target, sources
            print ""
            if not wrepeat:
                print str(i) + ' search condition sets'
                try:
                    w, ptc, cpaths = network.search_cit_components(sources, target, verbosity=verbosity)
                except:
                    print "Warning:!!!"
                    print network.search_cit_components(sources, target)
                    continue

                # Check whether w1 and ptnc are the same as w1old and ptncold
                # Get the parents not in the causal subgraph
                ptnc = list(set(pt) - set(ptc))
                # Get the maxdim maximum parents in w
                w1 = list(set(w) - set(ptnc))
                # Move w1old one step further
                w1old = [(wele[0], wele[1]-1) for wele in w1old]

                if set(w1old) == set(w1) and set(ptncold) == set(ptnc):
                    wrepeat = True

            else:
                print i
                # Get the parents not in the causal subgraph
                ptnc = list(set(pt) - set(ptc))
                # Get the maxdim maximum parents in w
                w1 = list(set(w) - set(ptnc))
                # Update w1
                w1 = [(wele[0], wele[1]-1) for wele in w1]
                # Combine w1 and ptnc
                w = w1 + ptnc

            # Reorganize the data
            data21   = reorganize_data(data, [target]+ptc+w)
            data22   = reorganize_data(data, [target]+w)
            # Drop the nan values
            data21   = dropna(data21)
            data22   = dropna(data22)
            datasize[i], dimsize[i] = data21.shape
            xyindex1 = [1,1+len(ptc)]
            if data21.shape[0] < 100 or data22.shape[0] < 100:
                print 'Not enough time series datapoints (<100)!'
                return citset, mitset, miset
            citset[i]  = computeCMIKNN(data21, k=k, xyindex=xyindex1) / np.log(base)
            pastset[i] = computeMIKNN(data22, k=k, xyindex=[1]) / np.log(base)

            sources = [(s[0], s[1]-1) for s in sources]

            if not wrepeat:
                # Copy w to wold
                wold = w
                # Get the parents not in the causal subgraph
                ptncold = list(set(pt) - set(ptc))
                # Get the maxdim maximum parents in w
                w1old = list(set(wold) - set(ptncold))

        return citset, pastset, titset, datasize, dimsize


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
