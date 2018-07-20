import os
# import pickle
import numpy as np

from info.core.construct_network_parallel import findCausalRelationships
from mpi4py import MPI
from mpi4py.MPI import Wtime

# Initialize MPI settings
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

###############################
# File and folder information #
###############################
file_name = os.path.expanduser('~') + '/codes/info/info/test_network.p'

##############
# Parameters #
##############
# Number of time series
N = 2000

# Kernel estimation
approach = 'knn'
k = 100

# Significance test
alpha  = 0.05
ntest  = 100
base   = np.e
deep   = True
contemp= True

# Convergence criteria
taumax, taumin = 4, 2
dtau = 2

# Variable names
varnames = ["X1", "X2", "X3"]

# Put all the parameters into a dictionary
paras = {"N": N, "approach":approach, "k":k,
         "alpha":alpha, "ntest":ntest, "base":base, "deep":deep, "contemp":contemp,
         "taumax":taumax, "taumin":taumin, "dtau":dtau,
         "vars":varnames}

####################
# Time series data #
####################
def trigauss(N):
    np.random.seed(1)
    data = np.random.randn(N, 3)
    for t in range(1, N):
        data[t, 1] += 0.6*data[t-1, 0]
        data[t, 2] += 0.6*data[t-1, 1] - 0.36*data[t-2, 0]
    return data

data = trigauss(N)

###############################
# Generate the causal network #
###############################
start = Wtime()
causalDict = findCausalRelationships(data, dtau=dtau, taumax=taumax, taumin=taumin, comm=comm,
                                     approach=approach, k=k, contemp=contemp,
                                     alpha=alpha, ntest=ntest, base=base, deep=deep)
if rank == 0:
    print "Time usage: ", Wtime()-start

####################
# Save the results #
####################
if rank == 0:
    import cPickle
    with open(file_name, 'wb') as f:
        cPickle.dump({"network":causalDict, "varnames":varnames, "data":data, "paras":paras}, f)
