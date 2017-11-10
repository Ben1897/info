"""
The script is for computing the information transfer based on the data and the
causal network.
"""

import numpy as np

from .info import info
from ..utils.causal_network import causal_network
from ..utils.others import reorganize_data

class info_network(object):

    def __init__(self, data, causalDict, taumax=10, kernel='gaussian', pdfest='kde_c', base=2.):
        """
        Input:
        data       -- the data [numpy array with shape (npoints, ndim)]
        causalDict -- dictionary of causal relationships, where
                      the keys are the variable at t time [int]
                      the values are the parents of the corresponding node [list of sets].
                      e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
        taumax     -- the maximum time lag for generating causal network [int]
        kernel     -- the kernel type [str]
        pdfest     -- the selected approach for computing PDF [string]
        base       -- the logrithmatic base [float]

        """
        self.data   = data
        self.taumax = int(taumax)
        self.pdfest = pdfest
        self.base   = float(base)
        self.kernel = kernel

        # Check whether the number of variables are larger than 1
        self.npoint, self.nvar = data.shape
        if self.nvar <= 1:
            raise Exception('Less than two nodes!')

        # Generate the causal network
        self.causalDict = causalDict
        self.network    = causal_network(causalDict, taumax)
        self.nnode      = self.network.nnodes
        if self.network.nvar != self.nvar:
            raise Exception('The numbers of variables in data and the causalDict are not equal!')

    def compute_2n_infotrans(self, source, target, conditioned=True, sidepath=True, causalpath=True, normalized=False, verbosity=1):
        """
        Compute the information transfer from a source node to a target node.

        Input:
        source      -- the source node [set (var_index, lag)]
        target      -- the target node [set (var_index, lag)]
        conditioned -- whether including conditions [bool]
        sidepath    -- whether including the contemporaneous sidepaths [bool]
        causalpath  -- whether computing MITP or MIT [bool]
        normalized  -- whether the calculated info metrics need to be normalized [bool]

        Ouput:
        an instance from the class info based on the method __computeInfo2D_conditioned() or __computeInfo2D()

        """
        data    = self.data
        network = self.network
        pdfest  = self.pdfest
        base    = self.base
        kernel  = self.kernel

        # If not conditioned, just compute the normal information metrics (not the momentary one)
        if not conditioned:
            # Reorganize the data
            data_required = reorganize_data(data, [source, target])
            # Compute the information transfer
            inforesult = info(ndim=2, data=data_required, approach=pdfest, bandwidth='silverman',
                              kernel=kernel, base=base, conditioned=False)
            # Normalize if necessary
            if normalized:
                inforesult.normalizeinfo()

            return inforesult

        # Check whether the two nodes are linked
        if network.check_links(source, target, verbosity=0) not in ['causalpath', 'directed', 'contemporaneous']:
            print "The source %s and the target %s are not linked through a causal path or a contemporaneous link!" % (source, target)
            return None

        # Generate the MIT/MITP conditions
        if causalpath:
            w = network.search_mitp_condition(source, target, sidepath=sidepath, verbosity=verbosity)
        else:
            w = network.search_mit_condition(source, target, sidepath=sidepath, verbosity=verbosity)

        # Reorganize the data
        data_required = reorganize_data(data, [source, target] + w)

        # Compute the information transfer
        if w:
            inforesult = info(ndim=2, data=data_required, approach=pdfest, bandwidth='silverman',
                              kernel=kernel, base=base, conditioned=True)
        else:
            inforesult = info(ndim=2, data=data_required, approach=pdfest, bandwidth='silverman',
                              kernel=kernel, base=base, conditioned=False)

        # Normalize if necessary
        if normalized:
            inforesult.normalizeinfo()

        return inforesult

    def compute_3n_infotrans(self, source1, source2, target, conditioned=True, sidepath=True, normalized=False, verbosity=1):
        """
        Compute the information transfer from two source nodes to a target node.

        Input:
        source1     -- the 1st source node [set (var_index, lag)]
        source2     -- the 2st source node [set (var_index, lag)]
        target      -- the target node [set (var_index, lag)]
        conditioned -- whether including conditions [bool]
        sidepath    -- whether including the contemporaneous sidepaths [bool]
        normalized  -- whether the calculated info metrics need to be normalized [bool]

        Ouput:
        an instance from the class info based on the method __computeInfo3D_conditioned() or __computeInfo3D()

        """
        data    = self.data
        network = self.network
        pdfest  = self.pdfest
        base    = self.base
        kernel  = self.kernel

        # If not conditioned, just compute the normal information metrics (not the momentary one)
        if not conditioned:
            # Reorganize the data
            data_required = reorganize_data(data, [source1, source2, target])
            # Compute the information transfer
            inforesult = info(ndim=3, data=data_required, approach=pdfest, bandwidth='silverman',
                              kernel=kernel, base=base, conditioned=False)
            # Normalize if necessary
            if normalized:
                inforesult.normalizeinfo()

            return inforesult

        # Check whether each source node is linked with the target through a causal path
        if network.check_links(source1, target, verbosity=verbosity) not in ['causalpath', 'directed']:
            print "The source %s and the target %s are not linked through a causal path!" % (source1, target)
            return None
        if network.check_links(source2, target, verbosity=verbosity) not in ['causalpath', 'directed']:
            print "The source %s and the target %s are not linked through a causal path!" % (source2, target)
            return None

        # Generate the MIT/MITP conditions
        w = network.search_mpid_condition(source1, source2, target, sidepath=sidepath, verbosity=verbosity)

        # Reorganize the data
        data_required = reorganize_data(data, [source1, source2, target] + w)
        # Compute the information transfer
        if w:
            inforesult = info(ndim=3, data=data_required, approach=pdfest, bandwidth='silverman',
                              kernel=kernel, base=base, conditioned=True)
        else:
            inforesult = info(ndim=3, data=data_required, approach=pdfest, bandwidth='silverman',
                              kernel=kernel, base=base, conditioned=False)

        # Normalize if necessary
        if normalized:
            inforesult.normalizeinfo()

        return inforesult

    def compute_2n_infotrans_set(self, source_ind, target_ind, conditioned=True, taumax=5, sidepath=True, causalpath=True, normalized=False,verbosity=1):
        """
        Compute the information transfer from a source node to a target node with lags varying from 1 to taumax

        Input:
        source_ind -- the source variable index [ind]
        target_ind -- the target variable index [ind]
        conditioned -- whether including conditions [bool]
        taumax     -- the maximum lag between the source node and the target
        sidepath   -- whether including the contemporaneous sidepaths [bool]
        causalpath  -- whether computing MITP or MIT [bool]
        normalized -- whether the calculated info metrics need to be normalized [bool]

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
                                                   sidepath=sidepath, normalized=normalized,
                                                   causalpath=causalpath, verbosity=verbosity)

        # Return the results
        return results

    def compute_3n_infotrans_set(self, source1_ind, source2_ind, target_ind, conditioned=True, taumax=5, sidepath=True, normalized=False, verbosity=1):
        """
        Compute the information transfer from two source nodes to a target node with lags varying from 1 to taumax

        Input:
        source1_ind -- the 1st source variable index [ind]
        source2_ind -- the 2nd source variable index [ind]
        target_ind -- the target variable index [ind]
        conditioned -- whether including conditions [bool]
        taumax     -- the maximum lag between the source node and the target
        sidepath   -- whether including the contemporaneous sidepaths [bool]
        normalized -- whether the calculated info metrics need to be normalized [bool]

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
                                                          sidepath=sidepath,
                                                          normalized=normalized,
                                                          verbosity=verbosity)

        # Return the results
        return results
