'''
Construct the time series graph for the network from observation data.
Because of the stationarity assumption, we return the parents of all the variables at a given time step.

Author: Peishi Jiang
Date: 2017-02-24

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
from ..utils.causal_network import get_node_number, intersect
from ..utils.sst import independence, conditionalIndependence


################
# Key Function #
################
def findCausalRelationships(data, dtau, taumax, taumin, deep=False,
                            ntest=100, sstmethod='traditional', kernel='gaussian', alpha=.05,
                            approach='kde_cuda_general', base=2., returnTrue=False):
    """
    Return the causal relationships among the variables or the parents of all the variables in a given time t in the time series graph.

    Inputs:
    data       -- the observation data [numpy array with shape (npoints, ndim)]
    dtau       -- the range of the time lags used for convergence check [int]
    taumax     -- the maximum time lag for updating parents (also used for convergence check) [int]
    taumin     -- the minimum time lag for updating parents (also used for convergence check) [int]
    deep       -- indicating whether to perform a deeper conditional independence test [bool]
    ntest      -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]

    """
    # Get the number of data points and the number of variables
    npts, ndim = data.shape

    # Initialize the graph, the parents set, and the time lag
    g          = nx.DiGraph(ndim=ndim, tau=0)  # the DAG for time series graph
    causalDict = {}            # the final parents set
    tau        = 0             # the time lag

    # Add all the nodes at tau = 0 to the graph g
    for j in range(ndim):
        nodedict    = (j, tau)
        node_number = get_node_number(nodedict, ndim, 0)
        g.add_node(node_number)

    # Update the parents at time lag tau
    while not isConvergence(g, taumin, taumax, dtau):
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

        print ""
        print "The preliminary parents ---"
        printParents(g, tau)

        # Get the parents Pa(Xj_tau) by excluding all the spurious parents based on the structures of the graph g, and update g accordingly
        for j in range(ndim):
            nodedict = (j, tau)  # The node of interest
            g, _     = excludeSpuriousParents(g, data, nodedict, deep=deep,
                                              ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                              approach=approach, base=base, returnTrue=returnTrue)

        # Print the current parents
        print ""
        print "The final parents ---"
        printParents(g, tau)

    # Once the graph converges, assign the parents of each node at the last time step to the parents set
    for j in range(ndim):
        nodedict      = (j, tau)    # the variable at the latest time
        paset         = getParents(g, (j, tau)) # TODO
        causalDict[j] = [(pa[0], pa[1]-tau) for pa in paset] # TODO

    # Return the parents set
    return causalDict


##################
# Help Functions #
##################
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
            parents = getParents(g, (j, tau))
            print "The parents of node %d:" % j
            print parents
        print ""

    return


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


def excludeSpuriousParents(g, data, nodedict, deep=False,
                           ntest=100, sstmethod='traditional', kernel='gaussian', alpha=.05,
                           approach='kde_cuda_general', base=2., returnTrue=False):
    """
    Exclude the spurious parents of the node of interest in the graph g.

    Inputs:
    g          -- the graph [graph]
    data       -- the observation data [numpy array with shape (npoints, ndim)]
    nodedict   -- the node of interest with format (index, tau) [tuple]
    deep       -- indicating whether to perform a deeper conditional independence test [bool]
    ntest      -- the number of the shuffles [int]
    sstmethod  -- the statistical significance test method [str]
    alpha      -- the significance level [float]
    kernel     -- the kernel used for KDE [str]
    approach   -- the package (cuda or c) used for KDE [str]
    base       -- the log base [float]
    returnTrue -- indicating whether the true mutual information is returned if the significant test fails [bool]

   """
    # Get the number of variables
    ndim = g.graph['ndim']

    # Get the node number for Xj_tau
    node_number = get_node_number(nodedict, ndim, 0)

    # Get the preliminary parents
    ppaset = set(g.predecessors(node_number))

    # Exclude the spurious parents
    for i in range(ndim):
        # Get the node number for Xi_0
        nodedict_p    = (i, 0)
        node_number_p = get_node_number(nodedict_p, ndim, 0)

        # Exclude Xi_0 if Xi_0 ind Xj_tau, and update g
        if independence(nodedict_p, nodedict, data,
                        ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                        approach=approach, base=base, returnTrue=returnTrue):
            print "Exclude the link %s -> %s:" % (nodedict_p, nodedict)
            ppaset.discard(node_number_p)
            g.remove_edge(node_number_p, node_number)

        # Get all the paths from Xi_0 to Xj_tau and the parents of Xj_tau in these paths
        nodes_in_paths = getCausalPaths(g, node_number_p, node_number)
        paseti         = set(intersect(nodes_in_paths, ppaset))
        paseti_dict    = getNodesDict(g, list(paseti))

        # Exclude Xi_0 if it is still in ppaset and the dependency Xi_0 -> Xj_tau is due to other paths
        if (node_number_p in paseti) and (len(paseti) > 1):
            if conditionalIndependence(nodedict_p, nodedict, data=data,
                                       conditionset=list(set(paseti_dict)-{nodedict_p}),
                                       ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                       approach=approach, base=base, returnTrue=returnTrue):
                print "Exclude the link %s -> %s conditioning on %s:" % (nodedict_p, nodedict, list(set(paseti_dict)-{nodedict_p}))
                ppaset.discard(node_number_p)
                paseti.discard(node_number_p)
                g.remove_edge(node_number_p, node_number)

        # Now, there are still more than parent of Xj_tau that are in the paths Xi_0 -> Xj_tau, check whether their links to Xj_tau are due to the common driver Xi_0
        if len(ppaset) > 1:
            for pa in paseti-{node_number_p}:
                pa_dict = getNodesDict(g, [pa])[0]
                if conditionalIndependence(pa_dict, nodedict, data=data,
                                           conditionset=[nodedict_p],
                                           ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                           approach=approach, base=base, returnTrue=returnTrue):
                    print "Exclude the link %s -> %s conditioning on %s:" % (nodedict_p, nodedict, [nodedict_p])
                    ppaset.discard(pa)
                    paseti.discard(pa)
                    g.remove_edge(pa, node_number)
                elif deep and conditionalIndependence(pa_dict, nodedict, data=data,
                                                      conditionset=list(set(paseti_dict)-{pa_dict}),
                                                      ntest=ntest, sstmethod=sstmethod, kernel=kernel, alpha=alpha,
                                                      approach=approach, base=base, returnTrue=returnTrue):
                    print "Exclude the link %s -> %s conditioning on %s:" % (nodedict_p, nodedict, list(set(paseti_dict)-{pa_dict}))
                    ppaset.discard(pa)
                    paseti.discard(pa)
                    g.remove_edge(pa, node_number)

    # Return the graph
    return g, ppaset


def isConvergence(g, taumin, taumax, dtau):
    """
    Check the convergence of the graph.

    Inputs:
    g      -- the graph [graph]
    dtau   -- the range of the time lags used for convergence check [int]
    taumax -- the maximum time lag for updating parents (also used for convergence check) [int]
    taumin -- the minimum time lag for updating parents (also used for convergence check) [int]

    """
    # Get the maximum lag in the graph
    tau, ndim = g.graph['tau'], g.graph['ndim']

    # Initialize the convergence
    convergence = True

    # Not meet the minimum lag requirement
    if tau < taumin:
        return False

    # Meet the maximum lag requirement
    if tau > taumax:
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
    pathall = list(pathall)

    # Unlist the pathall
    return [node for path in pathall for node in path]


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
