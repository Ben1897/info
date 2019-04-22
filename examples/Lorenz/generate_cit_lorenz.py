"""
This file is used for conducting the causal history analysis on the Lorenz model.
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
from info.models.others import Lorenz_model

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
N, lag = 3, 1
e, ez  = 1, 0.3
a = 4.
varnames = ['X', 'Y', 'Z']
T    = 13000
Tats = 10000
seed = 1
init = [0., 1., 1.05]  # the initial condition
s, r, b = 10., 28., 8./3.  # Lorenz parameters
dt = 0.01

# Define the corresponding parent set
causalDict = {0: [(0,-1), (1,-1)],
              1: [(0,-1), (1,-1), (2,-1)],
              2: [(0,-1), (1,-1), (2,-1)]}

para = {"T":Tats, "seed":seed, "init":init, "varnames": varnames, "taulag":[taustart, tauend],
        "taumax": taumax, "dt":dt,
        "s":s, "r":r, "b":b, "knn":knn, "causalDict":causalDict}

infofilename = './info_Lorenz.npy'

###############
# Compute CIT #
###############
info_to_save = {}
pid_to_save = {}
sst_to_save = {}
sizedim_to_save = {}

# Generate the data
data = Lorenz_model(N=T,seed=seed,dt=dt,e=e,init=init,s=s,r=r,b=b)
data = data[T-Tats:,:]

# Number of variables
npts, nvar = data.shape
print npts, nvar

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
what_to_save = {"info":infoset, "pid":pidset, "sstpast":sstpastset, "sizedim":sizedim_to_save,
                "transitive": transitive, "knn":knn, "para": para}
np.save(infofilename, what_to_save)
