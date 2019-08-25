import pickle
import copy
import numpy as np
import sys
sys.path.append('../..')

from info.utils.causal_network import causal_network
from info.core.info import info, computeCMIKNN, computeMIKNN
from info.core.info_network import info_network
from info.utils.others import reorganize_data, dropna
from info.utils.others import butter_filter, aggregate, normalize

##############
# Parameters #
##############
# Trange = [6000, 7000, 8000, 9000, 10000]
Trange = [200, 300, 400, 500, 600, 700, 800, 900, 1000,
          2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Trange = [200, 300, 400]

folder = './networks/'
filename = lambda T: 'logistic_cmiknn_parallel_30_T' +str(T)+ '_N4_full_shuffle_fullycoupled_ez02_results.pkl'

knn = 5
# taumax = 100
taumax = 20
transitive = True
causalApprox = False
approxDistant = False

taustart, tauend = 1, 50

approx = {'transitive':transitive, 'causalApprox': causalApprox, 'approxDistant':approxDistant}

####################################################
# Function for excluding contemporaneous neighbors #
####################################################
def exclude_contemporaneous(parents):
    parents_new = {}
    for node in parents.keys():
        parent_list = parents[node]
        parents_new[node] = []
        for parent in parent_list:
            if parent[1] != 0:
                parents_new[node].append(parent)
    return parents_new

###############
# Compute CIT #
###############
info_to_save = {}
pid_to_save = {}
sst_to_save = {}
sizedim_to_save = {}
for i in range(len(Trange)):
    # Load the parents and data
    dfn     = folder + filename(Trange[i])
    results = pickle.load(open(dfn, 'rb'))
    data    = results['fulldata'][0]

    # Normalize the data (for now)
    data = normalize(data)

    # Number of variables
    npts, nvar = data.shape

    # Coupling strengths and threshold
    parents  = results['results'][0]['parents_neighbors']
    parents_nocontemp = exclude_contemporaneous(parents)
    lagfuncs = results['results'][0]['parents_xy']

    # Create the template for the sources
    sources_template = [(j, -taustart) for j in range(nvar)]

    print dfn

    # Compute info metrics
    infoset = np.zeros([nvar, tauend-taustart, 5])
    pidset = np.zeros([nvar, tauend-taustart, 12])
    sstpastset = np.zeros([nvar, tauend-taustart, 4])
    if approxDistant and causalApprox:
        datasize = np.zeros([nvar, tauend-taustart, 5])
        dimsize = np.zeros([nvar, tauend-taustart, 5])
    elif causalApprox and not approxDistant:
        datasize = np.zeros([nvar, tauend-taustart, 4])
        dimsize = np.zeros([nvar, tauend-taustart, 4])
    elif not causalApprox and approxDistant:
        datasize = np.zeros([nvar, tauend-taustart, 4])
        dimsize = np.zeros([nvar, tauend-taustart, 4])
    elif not causalApprox and not approxDistant:
        datasize = np.zeros([nvar, tauend-taustart, 3])
        dimsize = np.zeros([nvar, tauend-taustart, 3])

    # Get the maximum lag of the parent
    paset = [pa for pset in parents_nocontemp.values() for pa in pset]
    mpl = np.max([-pr[1] for pr in paset])
    print mpl
    # Construct the network
    net = causal_network(parents_nocontemp, lagfuncs, taumax)

    # Construct the information network
    infonet = info_network(data, parents_nocontemp, lagfuncs=lagfuncs, taumax=taumax, causalApprox=causalApprox,
                           approach='knn', k=knn)

    # Compute the metrics
    for k in range(nvar):
        target = (k, 0)
        sources = copy.deepcopy(sources_template)
        results = infonet.compute_cit_set(sources, target, tauend-taustart, pidcompute=True,
                                          transitive=transitive, approxDistant=approxDistant, sst=True)
        # info from causal history
        infoset[k,:,0], infoset[k,:,1], infoset[k,:,2] = results['cit'], results['past'], results['tit']

        # PID from causal history
        pidset[k,:,:4], pidset[k,:,4:8], pidset[k,:,8:] = results['pidimmediate'], results['pidpast'], results['pidtotal']

        # Data size and dimensionality
        datasize[k,:,:], dimsize[k,:,:] = results['datasize'], results['dimsize']

        # SST
        sstpastset[k,:,:] = results['sstpast']

    sst_to_save[str(Trange[i])] = sstpastset
    info_to_save[str(Trange[i])] = infoset
    pid_to_save[str(Trange[i])] = pidset
    sizedim_to_save[str(Trange[i])] = {"dimsize":dimsize, "datasize":datasize}

####################
# Save the results #
####################
if transitive:
    infofilename = 'logistic_4var_anomaly_taumax5_transitive.pkl'
else:
    infofilename = 'logistic_4var_anomaly_taumax5.pkl'
pickle.dump({"info":info_to_save, "pid":pid_to_save, "sstpast":sst_to_save, "sizedim":sizedim_to_save, "approx":approx,
             "knn":knn, "taurange": (taustart,tauend)},open(folder+infofilename, 'wb'))