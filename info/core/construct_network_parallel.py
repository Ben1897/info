'''
Construct the time series graph for the network from observation data.
Because of the stationarity assumption, we return the parents of all the variables at a given time step.

Author: Peishi Jiang
Date: 2018-06-02

findCausalRelationships()

getPreliminaryParents()
updateGraphByParents()
excludeSpuriousParents()
isConvergence()
getParents()
temporallyMoveNodes()
getCausalPaths()
getNodesDict()

'''

import numpy as np
import networkx as nx
from pprint import pprint
# from ..utils.causal_network import get_node_number, intersect
from ..utils.sst import independenceParallel, conditionalIndependenceParallel, independence

import traceback
import sys

################
# Key Function #
################
def findCausalRelationships(data, dtau, taumax, taumin, comm, contemp=True, deep=False,
                            ntest=100, sstmethod='traditional', alpha=.05, approach='kde_cuda_general',
                            kernel='gaussian',  # parameters for KDE
                            k=5,                # parameters for KNN
                            base=2., returnTrue=False):
    """
    Return the causal relationships among the variables or the parents of all the variables in a given time t in the time series graph.

    Inputs:
    data       -- the observation data [numpy array with shape (npoints, ndim)]
    dtau       -- the range of the time lags used for convergence check [int]
    taumax     -- the maximum time lag for updating parents (also used for convergence check) [int]
    taumin     -- the minimum time lag for updating parents (also used for convergence check) [int]
    comm       -- the communicator for MPI
    contemp    -- indicating whether to include the direct contemporaneous links [bool]
    deep       -- indicating whether to perform a deeper conditional independence test [bool]
    ntest      -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]

    """
    # Get the number of the processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Check whether it is overestimated
    if rank == 0:
        if (ntest % size != 0):
            print "the number of processors must evenly divide the size of the vectors"
            comm.Abort()

    # Get the number of data points and the number of variables
    npts, ndim = data.shape

    # Initialize the graph, the final parents set, and the time lag
    g          = nx.DiGraph(ndim=ndim, tau=0)  # the DAG for time series graph
    causalDict = {}            # the final parents set
    tau        = 0             # the time lag

    # Add all the nodes and the contemporaneous links if required at tau = 0 to the graph g
    if rank == 0:
        print ""
        print "The contemporaneous parents ---"
    g = initializeNodesLinksAtLagZero(g, data, comm, contemp, approach=approach, kernel=kernel, k=k)

    if rank == 0:
        printParents(g, tau)

    # Update the parents at time lag tau
    while not isConvergence(g, taumin, taumax, dtau, comm):
        # Update the time lag
        tau += 1

        # Get the preliminary parents for Xj_tau, its preliminary parents PPa(Xj_tau),
        # and update the graph g by adding these parents
        for j in range(ndim):
            nodedict = (j, tau)  # The node of interest
            ppaset   = getPreliminaryParents(g, j)
            g        = updateGraphByParents(g, ppaset, nodedict)

        # Update the maximum lag in the graph
        g.graph['tau'] = tau

        if rank == 0:
            print ""
            print "The preliminary parents ---"
            printParents(g, tau)

        # Get the parents Pa(Xj_tau) by excluding all the spurious parents based on the structures of the graph g, and update g accordingly
        for j in range(ndim):
            nodedict = (j, tau)  # The node of interest
            g, _     = excludeSpuriousParents(g, data, nodedict, comm=comm, deep=deep,
                                              ntest=ntest, sstmethod=sstmethod, kernel=kernel, k=k, alpha=alpha,
                                              approach=approach, base=base, returnTrue=returnTrue)

        # Print the current parents
        if rank == 0:
            print ""
            print "The final parents ---"
            printParents(g, tau)

    # Once the graph converges, assign the parents of each node at the last time step to the parents set
    for j in range(ndim):
        nodedict      = (j, tau)    # the variable at the latest time
        paset         = getParents(g, (j, tau)) # TODO
        causalDict[j] = [(pa[0], pa[1]-tau) for pa in paset] # TODO

    # Print the network
    if rank == 0:
        print ""
        print "The network structure ---"
        pprint(causalDict)
        print ""

    # Return the parents set
    return causalDict


##################
# Help Functions #
##################
def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def union(alist):
    """ return the union of multiple lists """
    return set().union(*alist)


def exclude_intersection(a, b):
    """ return the subset of a which does not belong to b."""
    return [e for e in a if e not in intersect(a, b)]


def printParents(g, tau):
    """
    Print the parents of the nodes at time lag tau in the graph g.

    Inputs:
    g   -- the graph [graph]
    tau -- the time lag [int]

    """
    # Get the maximum time lag in g
    ndim, taumax = g.graph['ndim'], g.graph['tau']

    if tau > taumax:
        print "WARNING: the time lag %d is larger than the maximum lag %d in the graph." % (tau, taumax)
    else:
        print "Time lag: %d" % tau
        for j in range(ndim):
            target  = (j, tau)
            parents = getParents(g, target)
            print "The parents of node %s:" % (target,)
            print parents
        print ""

    return


def initializeNodesLinksAtLagZero(g, data, comm, contemp=True,
                                  ntest=100, sstmethod='traditional', kernel='gaussian', k=5, alpha=.05,
                                  approach='kde_cuda_general', base=2., returnTrue=False):
    """
    Initialize all the nodes and the contemporaneous links at tau=0.

    Inputs:
    g          -- the graph [graph]
    data       -- the observation data [numpy array with shape (npoints, ndim)]
    comm       -- the communicator for MPI
    contemp    -- indicating whether to include the direct contemporaneous links [bool]
    ntest      -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]
    """
    # Get the number of the processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get the maximum time lag and the number of variables in the graph g
    ndim, tau = g.graph['ndim'], g.graph['tau']

    if tau != 0:
        if rank == 0:
            raise Exception('The maximum lag in the graph is not zero!')
        comm.Abort()

    # Initialize the nodes at tau=0
    for j in range(ndim):
        nodedict    = (j, tau)
        node_number = get_node_number(nodedict, ndim, 0)
        g.add_node(node_number)

    # Include the contemporaneous links if necessary
    if contemp:
        for j in range(1, ndim):
            nodedict    = (j, tau)
            node_number = get_node_number(nodedict, ndim, 0)

            # Get the preliminary parents for Xj_0, which are [X1_0, ..., Xj-1_0]
            ppaset      = set(range(j))
            ppaset_dict = getNodesDict(g, list(ppaset))

            # Exclude the spurious parents Xi_0 in ppaset if
            # (1) Xi_0 ind Xj_0
            # (2) Xi_0 ind Xj_0 conditioning on the remaining nodes in ppaset
            node_number_remove = []
            for pa in ppaset:
                padict = getNodesDict(g, [pa])[0]

                # (1) Xi_0 ind Xj_0
                ind = independenceParallel(padict, nodedict, data, comm=comm, shuffle_ind=[0],
                                           ntest=ntest, sstmethod=sstmethod, kernel=kernel, k=k, alpha=alpha,
                                           approach=approach, base=base, returnTrue=returnTrue)
                if ind:
                    if rank == 0:
                        print "Exclude the link %s -> %s." % (padict, nodedict)
                    node_number_remove.append(pa)
                # (2) Xi_0 ind Xj_0 conditioning on the remaining nodes in ppaset
                else:
                    conditionset = list(set(ppaset_dict)-{padict})
                    if conditionset:
                        cond_ind = conditionalIndependenceParallel(padict, nodedict, data=data, comm=comm, shuffle_ind=[0],
                                                                   conditionset=conditionset,
                                                                   ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha, k=k,
                                                                   approach=approach, base=base, returnTrue=returnTrue)
                        if len(ppaset) > 1 and cond_ind:
                            if rank == 0:
                                print "Exclude the link %s -> %s conditioning on %s." % (padict, nodedict, list(set(ppaset_dict)-{padict}))
                            node_number_remove.append(pa)

            # Remove the spurious nodes
            paset =  ppaset - set(node_number_remove)

            # Link the remaining parents with the target node in the graph
            g = updateGraphByParents(g, paset, nodedict)

    return g


def getPreliminaryParents(g, j):
    """
    Get the preliminary parents for the variable j.

    Input:
    g -- the graph with the maximum lag tau [graph]
    j -- the index of the variable of interest Xj_tau+1 [int]
    """
    # Get the maximum lag and the number of variables in the graph
    tau, ndim = g.graph['tau'], g.graph['ndim']

    # Get the node number for both the node of interest and its previous node in the graph
    nodedict_now   = (j, tau+1) # Xj_tau+1
    nodedict_early = (j, tau)   # Xj_tau
    node_number_n  = get_node_number(nodedict_now, ndim, 0)
    node_number_e  = get_node_number(nodedict_early, ndim, 0)

    # Add Xj_tau+1 to the graph
    g.add_node(node_number_n)

    # Initialize an empty preliminary parent set
    ppaset = []

    # Get the parents of Xj_tau, Pa(Xj_tau, lags)
    ppaset1 = list(set(g.predecessors(node_number_e)))

    # Update Pa(Xj_tau, lags) to Pa(Xj_tau, lags+1)
    ppaset1 = temporallyMoveNodes(g, ppaset1, lag=1)

    # Add ppaset1 to ppaset
    ppaset += ppaset1

    # Add all the nodes at the earliest time to ppaset
    ppaset += range(ndim)

    # Return
    return ppaset


def updateGraphByParents(g, paset, nodedict):
    """
    Update the graph g by adding all the edges between the node of interest and its parents.

    Inputs:
    g        -- the graph [graph]
    paset    -- the parent set of the node of interest [list]
    nodedict -- the node of interest with format (index, tau) [tuple]

    """
    # Get the number of variables
    ndim = g.graph['ndim']

    # Get the node number
    node_number = get_node_number(nodedict, ndim, 0)

    # Add the edges
    for pa in paset:
        g.add_edge(pa, node_number)

    # Return
    return g


def excludeSpuriousParents(g, data, nodedict, comm, deep=False,
                           ntest=100, sstmethod='traditional', kernel='gaussian', alpha=.05,
                           approach='kde_cuda_general', k=5, base=2., returnTrue=False):
    """
    Exclude the spurious parents of the node of interest in the graph g.

    Inputs:
    g          -- the graph [graph]
    data       -- the observation data [numpy array with shape (npoints, ndim)]
    nodedict   -- the node of interest with format (index, tau) [tuple]
    comm       -- the communicator for MPI
    deep       -- indicating whether to perform a deeper conditional independence test [bool]
    ntest      -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]

    """
    # Get the number of the processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get the number of variables
    ndim = g.graph['ndim']

    # Get the node number for Xj_tau
    node_number = get_node_number(nodedict, ndim, 0)

    # Get the preliminary parents
    ppaset      = set(g.predecessors(node_number))
    ppaset_dict = getNodesDict(g, list(ppaset))

    # Exclude Xi_0 if Xi_0 ind Xj_tau, and update g
    for i in range(ndim):
        # Get the node number for Xi_0
        nodedict_p    = (i, 0)
        node_number_p = get_node_number(nodedict_p, ndim, 0)

        # Exclude Xi_0 if Xi_0 ind Xj_tau, and update g
        ind, mi, u, l = independenceParallel(nodedict_p, nodedict, data, comm=comm, shuffle_ind=[0],
                                             ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                             approach=approach, k=k, base=base, returnTrue=True)
        # # Code for checking whether independenceParallel is doing alright
        # if nodedict_p == (1,0) and nodedict == (0,2):
        #     if rank == 0:
        #         print ind, mi, u, l
        #         print independence(nodedict_p, nodedict, data, approach='knn', k=100, returnTrue=True)
        #         print "Check!!!"
        #         print ind
        #         comm.Abort()
        if ind:
            if rank == 0:
                print "Exclude the link %s -> %s." % (nodedict_p, nodedict)
            ppaset.discard(node_number_p)
            ppaset_dict.remove(nodedict_p)
            g.remove_edge(node_number_p, node_number)

    # Exclude Xi_0 if it is still in ppaset and Xi_0 ind Xj_tau given the remaining parents
    node_numbers_remove = []
    for i in range(ndim):
        # Get the node number for Xi_0
        nodedict_p    = (i, 0)
        node_number_p = get_node_number(nodedict_p, ndim, 0)

        # Exclude Xi_0 if it is still in ppaset and Xi_0 ind Xj_tau given the remaining parents
        if (node_number_p in ppaset) and (len(ppaset) > 1):
            cond_ind = conditionalIndependenceParallel(nodedict_p, nodedict, data=data, comm=comm, shuffle_ind=[0],
                                                       conditionset=list(set(ppaset_dict)-{nodedict_p}),
                                                       ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                                       approach=approach, k=k, base=base, returnTrue=returnTrue)
            # if rank == 0:
            #     print len(list(set(ppaset_dict)-{nodedict_p})) + 2
            # try:
            #     cond_ind = conditionalIndependenceParallel(nodedict_p, nodedict, data=data, comm=comm, shuffle_ind=[0],
            #                                                conditionset=list(set(ppaset_dict)-{nodedict_p}),
            #                                                ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
            #                                                approach=approach, k=k, base=base, returnTrue=returnTrue)
            # except:
            #     if rank == 0:
            #         print(traceback.format_exc())

            if cond_ind:
                if rank == 0:
                    print "Exclude the link %s -> %s conditioning on %s." % (nodedict_p, nodedict, list(set(ppaset_dict)-{nodedict_p}))
                node_numbers_remove.append(node_number_p)

    # Remove the nodes in the graph
    for node_number_start in set(node_numbers_remove):
        nodedict_start = getNodesDict(g, [node_number_start])[0]
        ppaset.discard(node_number_start)
        ppaset_dict.remove(nodedict_start)
        g.remove_edge(node_number_start, node_number)

    # Now, if there are still more than one parent of Xj_tau that are in the paths Xi_0 -> Xj_tau, check whether their links to Xj_tau are due to the common driver Xi_0
    node_numbers_remove = []
    for i in range(ndim):
        # Print the memory usage
        if rank == 0:
            import operator
            import os
            import psutil
            from itertools import islice
            memory = {}
            for var, obj in locals().items():
                memory[var] = sys.getsizeof(obj)
            sorted_m = sorted(memory.items(), key=operator.itemgetter(1), reverse=True)
            memory2 = list(islice(sorted_m, 5))
            print memory2
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss / 1024)

        # Get the node number for Xi_0
        nodedict_p    = (i, 0)
        node_number_p = get_node_number(nodedict_p, ndim, 0)

        # Get all the paths from Xi_0 to Xj_tau and the parents of Xj_tau in these paths
        # nodes_in_paths = getCausalPaths(g, node_number_p, node_number)
        # paseti         = set(intersect(nodes_in_paths, ppaset))
        # paseti_dict    = getNodesDict(g, list(paseti))
        paseti = set()
        if rank == 0:
            causalpaths = getCausalPaths(g, node_number_p, node_number)
            for path in causalpaths:
                paseti = paseti | set([node for node in intersect(path, ppaset) if node not in paseti])
        paseti = comm.bcast(paseti, root=0)
        paseti_dict = getNodesDict(g, list(paseti))

       # if len(ppaset) > 1:
        if len(paseti) > 1:
            for pa in paseti-{node_number_p}:
                pa_dict = getNodesDict(g, [pa])[0]

                cond_ind1 = conditionalIndependenceParallel(pa_dict, nodedict, data=data, comm=comm, shuffle_ind=[0],
                                                            conditionset=[nodedict_p],
                                                            ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                                            approach=approach, k=k, base=base, returnTrue=returnTrue)
                # if rank == 0:
                #     print 2 + len(nodedict_p)
                # try:
                #     cond_ind1 = conditionalIndependenceParallel(pa_dict, nodedict, data=data, comm=comm, shuffle_ind=[0],
                #                                                 conditionset=[nodedict_p],
                #                                                 ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                #                                                 approach=approach, k=k, base=base, returnTrue=returnTrue)
                # except:
                #     if rank == 0:
                #         print(traceback.format_exc())

                if cond_ind1:
                    if rank == 0:
                        print "Exclude the link %s -> %s conditioning on %s." % (pa_dict, nodedict, [nodedict_p])
                    node_numbers_remove.append(pa)
                # Check whether the link to Xj_tau is due to all the parents in paseti
                else:
                    if deep:
                        cond_ind2 = conditionalIndependenceParallel(pa_dict, nodedict, data=data, comm=comm, shuffle_ind=[0],
                                                                    conditionset=list(set(paseti_dict)-{pa_dict}),
                                                                    ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                                                    approach=approach, k=k, base=base, returnTrue=returnTrue)
                        # if rank == 0:
                        #     print 2 + len(nodedict_p)
                        # try:
                        #     cond_ind2 = conditionalIndependenceParallel(pa_dict, nodedict, data=data, comm=comm, shuffle_ind=[0],
                        #                                                 conditionset=list(set(paseti_dict)-{pa_dict}),
                        #                                                 ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                        #                                                 approach=approach, k=k, base=base, returnTrue=returnTrue)
                        # except:
                        #     if rank == 0:
                        #         print(traceback.format_exc())

                        if cond_ind2:
                            if rank == 0:
                                print "Exclude the link %s -> %s conditioning on %s." % (pa_dict, nodedict, list(set(paseti_dict)-{pa_dict}))
                            node_numbers_remove.append(pa)

    # Remove the nodes in the graph
    for node_number_start in set(node_numbers_remove):
        g.remove_edge(node_number_start, node_number)

    # Return the graph
    return g, ppaset


def isConvergence(g, taumin, taumax, dtau, comm):
    """
    Check the convergence of the graph.

    Inputs:
    g      -- the graph [graph]
    dtau   -- the range of the time lags used for convergence check [int]
    taumax -- the maximum time lag for updating parents (also used for convergence check) [int]
    taumin -- the minimum time lag for updating parents (also used for convergence check) [int]

    """
    # Get the number of the processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get the maximum lag in the graph
    tau, ndim = g.graph['tau'], g.graph['ndim']

    # Initialize the convergence
    convergence = True

    # Not meet the minimum lag requirement
    if tau < taumin:
        return False

    # Meet the maximum lag requirement
    if tau >= taumax:
        if rank == 0:
            print ""
            print "The maximum lag requirement for convergence is fulfilled..."
        return True

    # Meet the requiement for the consistent causal structure over time
    if tau <= dtau:  # when tau is too small to check
        return False
    else:           # when tau is large enough
        for i in range(dtau-1):
            for j in range(ndim):
                node1_dict, node2_dict = (j, tau-i), (j, tau-i-1)
                node_number_1 = get_node_number(node1_dict, ndim, 0)
                node_number_2 = get_node_number(node2_dict, ndim, 0)

                # Get the parents of Pa(Xj_tau-dtau) and Pa(Xj_tau-dtau-1)
                paset1 = g.predecessors(node_number_1)
                paset2 = g.predecessors(node_number_2)

                # Move Pa(Xj_tau_dtau-1) one lag forward
                paset2_moved = temporallyMoveNodes(g, paset2, lag=1)

                # Compare
                if set(paset1) != set(paset2_moved):
                    return False

    # Return
    if convergence:
        if rank == 0:
            print ""
            print "The parents consistancy for convergence is fulfilled..."
    return convergence


def temporallyMoveNodes(g, nodes, lag=1):
    """
    Temporally move the nodes in the graph according to the lag.

    Inputs:
    g     -- the graph [graph]
    nodes -- the list of node numbers in the graph g [list]
    lag   -- the time lag based on which the nodes are moved [int]
    """
    # Get the information of the graph
    tau, ndim = g.graph['tau'], g.graph['ndim']

    # Move the nodes
    nodes_moved = []
    for node in nodes:
        node_moved = node + lag*ndim
        nodes_moved.append(node_moved)

    # Return
    return nodes_moved


def getCausalPaths(g, snode, tnode):
    """
    Find the causal path from source node to the target node in graph g.

    Inputs:
    g     -- the graph [graph]
    snode -- the number of the source node [int]
    tnode -- the number of the target node [int]
    """
    # Get all the path from snode to tnode
    pathall = nx.all_simple_paths(g, snode, tnode)

    return pathall
    # pathall2 = list(pathall)

    # # Unlist the pathall
    # node_in_path = [node for path in pathall2 for node in path]

    # # Print the memory usage of the variables
    # print ""
    # print "Size of pathall: ", float(sys.getsizeof(pathall))/1024.
    # print "Size of pathall2: ", float(sys.getsizeof(pathall2))/1024.
    # print "Size of node_in_path: ", float(sys.getsizeof(node_in_path))/1024.
    # print "Length of node_in_path: ", len(node_in_path)
    # print ""

    # return node_in_path


def get_node_number(nodedict, nvar, lag):
    '''Convert a node in set version to the node number in the graph.'''
    return nodedict[0] + lag + nvar*abs(nodedict[1])


def getNodesDict(g, nodes_number):
    """
    Convert a list of nodes in number from graph g into the format with (index, tau).

    Inputs:
    g            -- the graph [graph]
    nodes_number -- the nodes in number [list]

    """
    # Get the graph information
    ndim = g.graph['ndim']

    # Return
    return [(node % ndim, node / ndim) for node in nodes_number]


def getParents(g, node_dict):
    """
    Get the parents of a node node_dict in the format with (index, tau).

    Inputs:
    g         -- the graph [graph]
    node_dict -- the node of interest with the format (index, tau) [tuple]

    """
    # Get the number of variables
    ndim = g.graph['ndim']

    # Get the node number
    node_number  = get_node_number(node_dict, ndim, 0)

    # Get the parents
    parents_number = g.predecessors(node_number)

    # Convert the parents in number to the dict format
    return getNodesDict(g, parents_number)

# def printMemoryUsage(nvar=5):
#     """
#     Print the memory usages of the first 5th largest variables.
#     """
#     pass
