"""
This file is used for conducting the causal history analysis on Ornstein-Uhlenbeck process.
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
from info.models.others import trivariate_OU

##############
# Parameters #
##############
knn = 5
taumax = 20
transitive = False
causalApprox = False
approxDistant = False
taustart, tauend = 1, 1001

# Model parameters
N = 3
dt = 0.01
seed = 1
init = [1.6, 1.0, 2.4]  # the initial condition
tet = np.array([[-0.5, 0.3,0],[.4, -0.4, -0.3],[0.4, 0.6, -0.7]], dtype='float')
rho = 1.
varnames = ['X', 'Y', 'Z']

T = 13000
Tats = 10000
Nt = 10000

# Define the corresponding parent set
causalDict = {0: [(0,-1), (1,-1)],
              1: [(0,-1), (1,-1), (2,-1)],
              2: [(0,-1), (1,-1), (2,-1)]}

para = {"N":N, "T":Tats, "varnames": varnames, "taulag":[taustart, tauend],
        "taumax": taumax, "seed":seed, "init":init, 
        "knn":knn, "dt":dt, "rho":rho,
        "causalDict":causalDict}

###############
# Compute CIT #
###############
info_to_save = {}
pid_to_save = {}
sst_to_save = {}
sizedim_to_save = {}

# Generate the data
data = trivariate_OU(N=Tats-1,seed=seed,init=init,dt=dt,tet=tet,rho=rho,trash=T-Tats)

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

sizedim_to_save = {"dimsize":dimsize, "datasize":datasize}

####################
# Save the results #
####################
infofilename = './info_OU.npy'

what_to_save = {"info":infoset, "pid":pidset, "sstpast":sstpastset, "sizedim":sizedim_to_save,
                "transitive": transitive, "knn":knn, "para": para}
np.save(infofilename, what_to_save)
#pickle.dump({"info":infoset, "pid":pidset, "sstpast":sstpastset, "sizedim":sizedim_to_save,
#             "transitive": transitive, "knn":knn, "para": para}, open(infofilename, 'wb'))
