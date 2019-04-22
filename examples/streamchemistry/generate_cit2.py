"""
This file is used for conducting causal history analysis for stream chemistry analysis example.
Case: flow rate corrected data.
"""
import pickle
import copy
import numpy as np
import sys
sys.path.append('../..')

from info.utils.causal_network import causal_network
from info.core.info import info, computeCMIKNN, computeMIKNN
from info.core.info_network import info_network
from info.utils.others import reorganize_data, dropna

##############
# Parameters #
##############
fns = ['parents_7hr_instream_strict_6var_SO4_anomaly_taumax5.pkl']

knn = 5
taumax = 20
prtypes = ['parents_nocontemp']
transitive = True
causalApprox = False
approxDistant = False

taustart, tauend = 1, 400

###############
# Compute CIT #
###############
info_to_save = {}
pid_to_save = {}
sst_to_save = {}
sizedim_to_save = {}
for i in range(len(fns)):
    # Load the parents
    prinfo = pickle.load(open(fns[i], 'rb'))
    prs = [prinfo['parents'][prtype] for prtype in prtypes]

    # Load the data
    dfn = prinfo['from']
    print dfn
    results = pickle.load(open(dfn, 'rb'))
    data    = results['fulldata'][0]

    # Number of variables
    npts, nvar = data.shape

    # Coupling strengths and threshold
    lagfuncs=results['results'][0]['parents_xy']

    # Create the template for the sources
    sources_template = [(j, -taustart) for j in range(nvar)]

    # print dfn, fns[i]

    # Compute info metrics
    infoset = np.zeros([len(prtypes), nvar, tauend-taustart, 5])
    pidset = np.zeros([len(prtypes), nvar, tauend-taustart, 12])
    sstpastset = np.zeros([len(prtypes), nvar, tauend-taustart, 4])
    if approxDistant and causalApprox:
        datasize = np.zeros([len(prtypes), nvar, tauend-taustart, 5])
        dimsize = np.zeros([len(prtypes), nvar, tauend-taustart, 5])
    elif causalApprox and not approxDistant:
        datasize = np.zeros([len(prtypes), nvar, tauend-taustart, 4])
        dimsize = np.zeros([len(prtypes), nvar, tauend-taustart, 4])
    elif not causalApprox and approxDistant:
        datasize = np.zeros([len(prtypes), nvar, tauend-taustart, 4])
        dimsize = np.zeros([len(prtypes), nvar, tauend-taustart, 4])
    elif not causalApprox and not approxDistant:
        datasize = np.zeros([len(prtypes), nvar, tauend-taustart, 3])
        dimsize = np.zeros([len(prtypes), nvar, tauend-taustart, 3])

    for j in range(len(prtypes)):
        print fns[i] + ' -- ' + prtypes[j]
        # Get the maximum lag of the parent
        paset = [pa for pset in prs[j].values() for pa in pset]
        mpl = np.max([-pr[1] for pr in paset])
        print mpl
        # Construct the network
        net = causal_network(prs[j], lagfuncs, taumax)

        # Construct the information network
        infonet = info_network(data, prs[j], lagfuncs=lagfuncs, taumax=taumax, causalApprox=causalApprox,
                               approach='knn', k=knn)

        # Compute the metrics
        for k in range(nvar):
            target = (k, 0)
            sources = copy.deepcopy(sources_template)
            results = infonet.compute_cit_set(sources, target, tauend-taustart, pidcompute=True,
                                              transitive=transitive, approxDistant=approxDistant, sst=True)
            # info from causal history
            infoset[j,k,:,0], infoset[j,k,:,1], infoset[j,k,:,2] = results['cit'], results['past'], results['tit']
            # infoset[j,k,:,3], infoset[j,k,:,4] = results['pasto'], results['pasta']

            # PID from causal history
            pidset[j,k,:,:4], pidset[j,k,:,4:8], pidset[j,k,:,8:] = results['pidimmediate'], results['pidpast'], results['pidtotal']

            # Data size and dimensionality
            datasize[j,k,:,:], dimsize[j,k,:,:] = results['datasize'], results['dimsize']

            # SST
            sstpastset[j,k,:,:] = results['sstpast']

    sst_to_save[fns[i]] = sstpastset
    info_to_save[fns[i]] = infoset
    pid_to_save[fns[i]] = pidset
    sizedim_to_save[fns[i]] = {"dimsize":dimsize, "datasize":datasize}

####################
# Save the results #
####################
infofilename = 'info_instream_new_6var_SO4_anomaly_transred_taumax5.pkl'
pickle.dump({"info":info_to_save, "pid":pid_to_save, "sstpast":sst_to_save, "sizedim":sizedim_to_save,
             "transitive": transitive, "prtype":prtypes, "knn":knn, "taurange": (taustart,tauend)}, open(infofilename, 'wb'))