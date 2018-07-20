from info.utils.sst import *
import numpy as np

from mpi4py import MPI
from mpi4py.MPI import Wtime

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters
ntest = 100
alpha = 0.05
approach = 'knn'
base  = np.e
k = 100
returnTrue = True
T = 2000
option = 'mi'

if option == 'mi':
    # Generate the sampled data
    rho  = 0.9
    mean = [0,0]
    cov  = [[1,rho],[rho,1],]
    data = np.random.multivariate_normal(mean, cov, T)
    node1, node2 = (0,0), (1,0)

    start = Wtime()
    result1, mi1, upper1, lower1 = independenceParallel(node1, node2, data, comm,
                                                        ntest=ntest, alpha=alpha,
                                                        approach=approach, base=base,
                                                        k=k, returnTrue=returnTrue)
    time1 = Wtime() - start

    if rank == 0:
        # start = Wtime()
        # result2, mi2, upper2, lower2 = independence(node1, node2, data,
        #                                             ntest=ntest, alpha=alpha,
        #                                             approach=approach, base=base,
        #                                             k=k, returnTrue=returnTrue)
        # time2 = Wtime() - start

        start = Wtime()
        result3, mi3, upper3, lower3 = independence(node1, node2, data,
                                                     ntest=ntest, alpha=alpha,
                                                     approach='kde_cuda_general', base=base,
                                                     bandwidth='silverman', sstmethod='traditional', kernel='gaussian',  # Parameters for KDE
                                                     returnTrue=returnTrue)
        time3 = Wtime() - start

        print mi1, upper1, lower1, result1
        # print mi2, upper2, lower2, result2
        print mi3, upper3, lower3, result3
        print time1, time3

elif option == 'cmi':
    # Generate the sampled data
    rho1, rho2, rho3 = .0, 0.4, 0.8
    mean = [0,0,0]
    cov  = [[1,rho1,rho3],[rho1,1,rho2],[rho3,rho2,1]]
    data = np.random.multivariate_normal(mean, cov, T)
    node1, node2 = (0,0), (1,0)
    conditionset = [(2,0)]

    start = Wtime()
    result1, cmi1, upper1, lower1 = conditionalIndependenceParallel(node1, node2, conditionset, data, comm,
                                                                   ntest=ntest, alpha=alpha,
                                                                   approach=approach, base=base,
                                                                   k=k, returnTrue=returnTrue)
    time1 = Wtime() - start

    if rank == 0:
        start = Wtime()
        result2, cmi2, upper2, lower2 = conditionalIndependence(node1, node2, conditionset, data,
                                                                ntest=ntest, alpha=alpha,
                                                                approach=approach, base=base,
                                                                k=k, returnTrue=returnTrue)
        time2 = Wtime() - start

        start = Wtime()
        result3, cmi3, upper3, lower3 = conditionalIndependence(node1, node2, conditionset, data,
                                                                ntest=ntest, alpha=alpha,
                                                                approach='kde_cuda_general', base=base,
                                                                bandwidth='silverman', sstmethod='traditional', kernel='gaussian',  # Parameters for KDE
                                                                returnTrue=returnTrue)
        time3 = Wtime() - start

        print cmi1, upper1, lower1, result1
        print cmi2, upper2, lower2, result2
        print cmi3, upper3, lower3, result3
        print time1, time2, time3
