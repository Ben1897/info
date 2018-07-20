import pickle
import numpy as np

from info.core.construct_network_parallel import findCausalRelationships
from mpi4py import MPI
from mpi4py.MPI import Wtime

# Initialize MPI settings
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# Parameters
N = 1000

# Kernel estimation
approach = 'knn'
k = 100

# Significance test
alpha  = 0.05
ntest  = 100
base   = 2.
deep   = True
contemp=True
model = 'trigauss'

# Convergence criteria
taumax, taumin = 4, 2
dtau = 2

def blgauss(N,a,b):
    np.random.seed(1)
    xn = lambda x,y: a*x + np.random.normal()
    yn = lambda x,y: b*x + np.random.normal()

    x0, y0 = np.random.normal(), np.random.normal()
    trash = 1000
    d = np.zeros([N+trash,2])
    d[0,0], d[0,1] = x0, y0
    for i in range(1,N+trash):
        d[i,0], d[i,1] = xn(d[i-1,0],d[i-1,1]), yn(d[i-1,0],d[i-1,1])

    return d[trash:,:]

def trigauss(N):
    np.random.seed(1)
    data = np.random.randn(N, 3)
    for t in range(1, N):
        data[t, 1] += 0.6*data[t-1, 0]
        data[t, 2] += 0.6*data[t-1, 1] - 0.36*data[t-2, 0]
    return data

def quadgauss(N):
    np.random.seed(1)
    trash = 2000
    data = np.random.randn(N+trash, 4)
    for t in range(1, N+trash):
        data[t, 0] += 0.7*data[t-1, 0] - 0.8*data[t-1, 1]
        data[t, 1] += 0.8*data[t-1, 1] + 0.8*data[t-1, 3]
        data[t, 2] += 0.5*data[t-1, 2] + 0.5*data[t-2, 1] + 0.6*data[t-3, 3]
        data[t, 3] += 0.7*data[t-1, 3]
    return data[trash:]


if model == 'blgauss':
    a, b = .3, .5
    varnames = ['X', 'Y']

    data = blgauss(N,a,b)

    start = Wtime()
    causalDict = findCausalRelationships(data, dtau=dtau, taumax=taumax, taumin=taumin, comm=comm,
                                         approach=approach, k=k, contemp=contemp,
                                         alpha=alpha, ntest=ntest, base=base, deep=deep)
    if rank == 0:
        print "Time usage: ", Wtime()-start

elif model == 'trigauss':
    data = trigauss(N)
    varnames = ['X1', 'X2', 'X3']

    start = Wtime()
    causalDict = findCausalRelationships(data, dtau=dtau, taumax=taumax, taumin=taumin, comm=comm,
                                         approach=approach, k=k, contemp=contemp,
                                         alpha=alpha, ntest=ntest, base=base, deep=deep)
    if rank == 0:
        print "Time usage: ", Wtime()-start

elif model == 'quadgauss':
    # Convergence criteria
    taumax, taumin = 10, 4
    dtau = 6

    start = Wtime()
    causalDict = findCausalRelationships(data, dtau=dtau, taumax=taumax, taumin=taumin, comm=comm,
                                         approach=approach, k=k, contemp=contemp,
                                         alpha=alpha, ntest=ntest, base=base, deep=deep)
    if rank == 0:
        print "Time usage: ", Wtime()-start

# if comm.Get_rank() == 0:
#     g = create_ts_graph(causalDict, varnames, lagmax=taumax, highlightednodes=[],
#                         highlightededges=[], engine='neato')
#     g
