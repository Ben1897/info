"""
This script is used for parsing the estimated network from running TIGRAMITE in terms of
(1) the pc algorithm results
(2) the lagfuncs and the sigthres

class tigramite_network()
    __init__()
    __get_MIT_network()
    __intersect_MIT_PC()
    get_network()
    update_truenetwork()
    intersect_two_networks()
    filter_network()
    compute_DR_FPR()
    convert_network_style()
    plot()

"""

import numpy as np

class tigramite_network(object):

    def __init__(self, PCnet, lagfuncs, sigthres, absolute=True):
        """
        Input:
        PCnet    -- dictionary of causal relationships from the PC algorithm, where
                    the keys are the variable at t time [int]
                    the values are the parents of the corresponding node [list of sets].
                    e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
        lagfuncs -- the momentary information transfer estimated from causalDict
                    [numpy array with shape (nvar, nvar, taumax+1)]
        sigthres -- the significance threshold of MIT
                    [numpy array with shape (nvar, nvar, taumax+1)]
        absolute -- whether taking the absolute values of lagfuncs and sigthres [bool]

        """
        self.PCnet    = PCnet
        if absolute:
            self.lagfuncs = np.abs(lagfuncs)
            self.sigthres = np.abs(sigthres)
        else:
            self.lagfuncs = lagfuncs
            self.sigthres = sigthres

        # Get the variable info
        self.vars = self.PCnet.keys()
        self.nvar = len(self.vars)

        # Check the shapes
        lagshape = lagfuncs.shape
        if lagshape != sigthres.shape:
            raise Exception("The shapes between lagfuncs and sigthres are not identical!")
        if lagshape[0] != self.nvar or lagshape[1] != self.nvar:
            raise Exception("The number of variables in lagfuncs is not identical to that in causalDict!")
        self.taumax = lagshape[2] - 1

        # Generate the network based on MIT results
        self.__get_MIT_network()

        # Generate the intersection between the MIT result and the PC result
        self.__intersect_MIT_PC()

        # Convert PC, MIT and PCMIT network into numpy array format
        self.PCnetn    = self.convert_network_style(network='pc')
        self.MITnetn   = self.convert_network_style(network='mit')
        self.PCMITnetn = self.convert_network_style(network='pcmit')

        # Initialize the true network
        self.truenet  = None
        self.truenetn = None

        # # Compute the threshold with 1C1D condition and 1D condition
        # self.sigthres_1c1d = filter_sigthres(lagfuncs, sigthres, choices='1C1D')
        # self.sigthres_1d   = filter_sigthres(lagfuncs, sigthres, choices='1D')

    def __get_MIT_network(self):
        """
        Return the MIT network based on lagfuncs and sigthres.
        """
        var, nvar, taumax  = self.vars, self.nvar, self.taumax
        lagfuncs, sigthres = self.lagfuncs, self.sigthres

        MITnet = {}
        lags = np.arange(taumax+1)
        for i in range(nvar):
            # For each variable, initialize a list for its neighbors and parents
            MITnet[i] = []
            # Get the parents or neighbors whose lagfuncs are larger than sigthres
            for j in range(nvar):
                lagfs, thres = lagfuncs[j, i, :], sigthres[j, i, :]
                # Get the index of significant lags
                # index = np.where(lagfs > thres)[0]
                # if i == j and 0 in index:  # If it is self-driving, the zero-lag is omitted
                #     index = index[1:]

                # Get the significant lags
                drivinglags = lags[np.where(lagfs > thres)]
                MITnet[i]  += [(j, -k) for k in drivinglags]
                # for k in lags[np.where(lagfs > thres)]:
                #     MITnet[i] += [(j, -k)]

        self.MITnet = MITnet

    def __intersect_MIT_PC(self):
        """
        Return the network which is the intersection between PC and MIT networks.
        """
        self.PCMITnet = self.intersect_two_networks('PC', 'MIT')

    def get_network(self, network='pc'):
        """
        Get the network.
        """
        if network.lower() == 'pc':
            return self.PCnet
        elif network.lower() == 'mit':
            return self.MITnet
        elif network.lower() == 'pcmit':
            return self.PCMITnet
        elif network.lower() == 'true':
            if self.truenet is None:
                raise Exception('Update the true network first!')
            return self.truenet

    def update_truenetwork(self, truenetwork):
        """
        Update the true causal network.

        Input:
        truenetwork -- dictionary of causal relationships from the PC algorithm, where
                       the keys are the variable at t time [int]
                       the values are the parents of the corresponding node [list of sets].
                       e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
        """
        self.truenet  = truenetwork
        self.truenetn = self.convert_network_style(network='true')

    def intersect_two_networks(self, network1='pc', network2='mit'):
        """
        Return the network which is the intersection between network1 and network2.
        """
        # Get the network from the class if it is PC, MIT or PCMIT network
        if isinstance(network1, str):
            net1 = self.get_network(network1)
        else:
            net1 = network1
        if isinstance(network2, str):
            net2 = self.get_network(network2)
        else:
            net2 = network2

        # Get the intersection between net1 and net2
        varnames = net1.keys()
        newnet   = dict.fromkeys(varnames)
        for varn in varnames:
            pcdep, pcmcdep = net1[varn], net2[varn]
            newnet[varn] = list(set(pcdep) & set(pcmcdep))
        return newnet

    def filter_network(self, network='pc', lagfuncs=None, contemp=True):
        """
        Filter the network by ensuring that a target variable has at most two contributions towards a source variable
        (i.e., one directed link with the most link strength and one contemporaneous link if contemp is True).

        Input:
        network -- either a string referring to the existing network in the class,
                   or dictionary of causal relationships from the PC algorithm, where
                   the keys are the variable at t time [int]
                   the values are the parents of the corresponding node [list of sets].
                   e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
        contemp -- whether including the contemporaneous node
        choice  -- the filtering choices
                   '1C1D': a target variable has at most two contributions towards a source variable
                             (i.e, one contemporaneous link and one directed link with the most link strength).
                   '1D'  : a target variable has at most one contribution towards a source variable
                             (i.e, one directed link with the most link strength).

        """
        # Get the network from the class if it is PC, MIT or PCMIT network
        if isinstance(network, str):
            net = self.get_network(network)
        else:
            net = network

        # Get the lagfuncs
        if lagfuncs is None:
            lagfuncs = self.lagfuncs

        # Filtering
        nvar   = self.nvar
        newnet = {}
        for i in range(nvar):
            newnet[i] = []
            drivings  = net[i]
            for j in range(nvar):
                # Get the force from j to i
                driving_itoj = [d for d in drivings if d[0] == j]
                # Get the index of the direct link
                driving_itoj_d_ind = [-d[1] for d in driving_itoj if d[1] != 0]
                # Get the index of the contemporaneous link
                driving_itoj_c_ind = [-d[1] for d in driving_itoj if d[1] == 0]
                # Get the lag with the maximum lagfunc value
                if driving_itoj_d_ind != []:
                    lags                      = lagfuncs[j, i, driving_itoj_d_ind]
                    maxlagind                 = np.argmax(lags)
                    driving_itoj_d_ind_unique = driving_itoj_d_ind[maxlagind]
                else:
                    driving_itoj_d_ind_unique = None

                # Assemble newnet[i]
                if contemp and driving_itoj_c_ind != []:
                    newnet[i] += [(j, 0)]
                if driving_itoj_d_ind_unique is not None:
                    newnet[i] += [(j, -driving_itoj_d_ind_unique)]

                # # Get the contemporaneous driving from j to i
                # driving_itoj_c = [d for d in drivings if d[0] == j and d[1] == 0]
                # # Get the direct driving from j to i
                # driving_itoj_d = [d for d in drivings if d[0] == j and d[1] != 0]
                #
                # lags      = np.array([lagfuncs[j, i, -d[1]] for d in driving_itoj_d])
                # maxlagind =

        return newnet

    def compute_DR_FPR(self, true_network='true', est_network='pc'):
        """
        Compute the performance of est_network compared with true_network.
        Return the detection rate and the false positive rate.
        """
        # Get the network from the class if it is PC, MIT or PCMIT network
        if isinstance(true_network, str):
            true_net = self.get_network(true_network)
        else:
            true_net = true_network
        if isinstance(est_network, str):
            est_net  = self.get_network(est_network)
        else:
            est_net  = est_network

        nvar, tau_max  = self.nvar, self.taumax
        tp, fp, tn, fn = 0., 0., 0., 0.
        l = true_net.values()  # all the drivings
        N = len([item for sublist in l for item in sublist])  # the number of all the drivings
        for i in range(nvar):
            true_causes       = set(true_net[i])
            est_causes        = set(est_net[i])
            right_causes      = set(true_causes) & set(est_causes)
            false_causes      = est_causes - right_causes     # false_positive
            undetected_causes = true_causes - right_causes

            # calculate # of true_positive, false_positive, true_negative, false_negative
            tp += len(right_causes)
            fp += len(false_causes)
            tn += (tau_max + 1)*nvar - 1 - len(true_causes | est_causes)
            fn += len(undetected_causes)

        # Compute FPR and DR
        dr, fpr = tp / N, fp / (fp+tn)

        return dr, fpr

    def convert_network_style(self, network='pc', taumax=None):
        """Convert the style of the network from a dictionary to a numpy array."""
        # Get the network from the class if it is PC, MIT or PCMIT network
        if isinstance(network, str):
            net = self.get_network(network)
        else:
            net = network
        if taumax is None:
            taumax = self.taumax

        # Conver the network in numpy array format
        varnames = net.keys()
        nvar     = len(varnames)
        networkn = np.zeros((nvar, nvar, taumax+1), dtype=bool)
        for i in range(nvar):
            target = varnames[i]
            for depend in net[target]:
                source, lag = depend[0], -depend[1]
                networkn[source, target, lag] = True

        # Get an 'artificial' lagfuncs for networkn for plotting
        lagfuncs = np.copy(self.lagfuncs)
        lagfuncs[np.logical_not(networkn)] = 0.

        return networkn, lagfuncs

    def plot(self):
        pass


## Help functions

# def filter_sigthres(lagfuncs, sigthres, choices='1C1D'):
#     """
#     Filter the network based on different filtering choices.
#
#     Input:
#     lagfuncs -- the momentary information transfer estimated from causalDict
#                 [numpy array with shape (nvar, nvar, taumax+1)]
#     sigthres -- the significance threshold of MIT
#                 [numpy array with shape (nvar, nvar, taumax+1)]
#     choice   -- the filtering choices
#                 '1C1D': a target variable has at most two contributions towards a source variable
#                           (i.e, one contemporaneous link and one directed link with the most link strength).
#                 '1D'  : a target variable has at most one contribution towards a source variable
#                           (i.e, one directed link with the most link strength).
#
#     """
#     shape        = sigthres.shape
#     sigthres_new = np.ones(shape)  # Let all the new sigthres be one
#     nvar = shape[0]
#     for i in range(nvar):
#         for j in range(nvar):
#             lagfs, thres = lagfuncs[j, i, :], sigthres[j, i, :]
#             # The contemporaneous undirected link
#             if i != j and lagfs[0] >= thres[0]:
#                 sigthres_new[j, i, 0] = thres[0]
#             # The directed link
# #             index = np.where(np.abs(lagfs) >= thres)[0]
#             index = np.where(lagfs < thres)[0]
#             # Include the lag-zero if it is self-dependence
#             if i == j and 0 not in index:
# #                 index = np.concatenate(index, [0])
#                 index += [0]
#             # Make all the insignificant zeros
#             lagfs[index] = 0.
#             if index.size:
#                 dl = np.argmax(lagfs)
# #                 if j == 3 and i == 1:
# #                     print lagfs
#                 sigthres_new[j, i, dl] = thres[dl]
