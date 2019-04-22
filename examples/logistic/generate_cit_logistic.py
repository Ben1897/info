"""
This file is used for conducting the causal history analysis on a trivariate logistic model.
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
from info.models.logistic_network import Logistic

##############
# Parameters #
##############
knn = 5
taumax = 20
transitive = False
causalApprox = False
approxDistant = False
taustart, tauend = 1, 51

# Model parameters
N, lag = 3, 1
e, ez  = 1, 0.3
a = 4.
varnames = ['X1', 'X2', 'X3']
T    = 13000
Tats = 10000
noiseType = 'additive'
noiseDist = 'uniform'
noisePara = [1, 0, 1]
adjM = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
lagM = np.array([[lag, lag, lag], [lag, lag, lag],
                    [lag, lag, lag]])

# Define the corresponding parent set
causalDict = {0: [(0,-1), (1,-1), (2,-1)],
              1: [(0,-1), (1,-1), (2,-1)],
              2: [(0,-1), (1,-1), (2,-1)]}

para = {"T":Tats, "varnames": varnames, "taulag":[taustart, tauend],
        "taumax": taumax,
        "knn":knn, "e":e, "ez":ez, "a":a,
        "noiseType":noiseType, "noiseDist":noiseDist, "noisePara":noisePara,
        "causalDict":causalDict, "adjM":adjM, "lagM":lagM}

###############
# Compute CIT #
###############
info_to_save = {}
pid_to_save = {}
sst_to_save = {}
sizedim_to_save = {}

# Generate the data
logistics = Logistic(N, adjM, lagM, e, ez, a=a, noiseType=noiseType,
                     noiseDist=noiseDist, noisePara=noisePara)
data = logistics.simulate(T).T
data = data[T-Tats:,:]

# Number of variables
npts, nvar = data.shape

# Create the template for the sources
sources_template = [(j, -taustart) for j in range(nvar)]

# Compute info metrics
infoset    = np.zeros([nvar, tauend-taustart, 5])
pidset     = np.zeros([nvar, tauend-taustart, 12])
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

# Construct the network
# net = causal_network(causalDict, taumax=taumax)

# Construct the information network
infonet = info_network(data, causalDict, taumax=taumax, causalApprox=causalApprox,
                       approach='knn', k=knn)

# Compute the metrics
target = (0, 0)
sources = copy.deepcopy(sources_template)
results = infonet.compute_cit_set(sources, target, tauend-taustart, pidcompute=True,
                                  transitive=transitive, approxDistant=approxDistant, sst=True)
    # info from causal history
for k in range(nvar):
    infoset[k,:,0], infoset[k,:,1], infoset[k,:,2] = results['cit'], results['past'], results['tit']

    # PID from causal history
    pidset[k,:,:4], pidset[k,:,4:8], pidset[k,:,8:] = results['pidimmediate'], results['pidpast'], results['pidtotal']

    # Data size and dimensionality
    datasize[k,:,:], dimsize[k,:,:] = results['datasize'], results['dimsize']

    # SST
    sstpastset[k,:,:] = results['sstpast'] 

sizedim_to_save = {"dimsize":dimsize, "datasize":datasize}

####################
# Save the results #
####################
infofilename = './simulated_results/info_logistic_noise03.npy'

what_to_save = {"info":infoset, "pid":pidset, "sstpast":sstpastset, "sizedim":sizedim_to_save,
                "transitive": transitive, "knn":knn, "para": para}
np.save(infofilename, what_to_save)
#pickle.dump({"info":infoset, "pid":pidset, "sstpast":sstpastset, "sizedim":sizedim_to_save,
#             "transitive": transitive, "knn":knn, "para": para}, open(infofilename, 'wb'))