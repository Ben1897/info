'''
This script is used for generating the condition of a MIP given:
(1) the causality relationship between different variables;
(2) the two sources and the target

Ref:
Jiang and Kumar PRE, in preparation (2017)

'''

import networkx as nx


def search_mpid_condition(causalDict, source1, source2, target, taumax=4, verbosity=1):
    '''
    Generate the condition of a MIP.

    Input:
    causalDict -- dictionary of causal relationships, where
                  the keys are the variable at t time [int]
                  the values are the parents of the corresponding node [list of sets].
                  e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
    source1 -- the first source node [set]
    source2 -- the second source node [set]
    target -- the target node [set]
    taumax -- the maximum time lag [int]
    verbosity -- the printing level [int]

    Ouput:
    pt -- the parents of the target node [list of sets]
    s1path -- the path from the first source node to the target node [list of sets]
    s2path -- the path from the second source node to the target node [list of sets]
    ps1path -- the parents of s1path [list of sets]
    ps2path -- the parents of s2path [list of sets]
    w -- the condition of the two causal paths [list of sets]
    w1 -- the condition of the first causal path [list of sets]
    w2 -- the condition of the second causal path [list of sets]
    '''
    # Convert causalDict to a dictionary with integer keys
    # mapvar, causalDictInt = convert_causalDict_to_int(causalDict)
    if verbosity > 0:
        print '------- The two sources are:'
        print source1, source2
        print '------- The target is:'
        print target
        print ''

    # Create an empty directed graph
    g = nx.DiGraph()

    # Assign all the nodes into the directed graph
    var = causalDict.keys()
    nvar = len(var)
    nnodes = nvar*(taumax+1)
    g.add_nodes_from(range(nnodes))

    # Assign all the edges
    for i in range(taumax+1):
        gap = nvar*i
        involed_nodes = range(gap, gap+nvar)
        for j in range(nvar):
            end = involed_nodes[j]
            for parent in causalDict[j]:
                start = get_node_number(parent, nvar, gap)
                g.add_edge(start, end)
    # print g.edges()

    # Generate pt, s1path, s2path
    tnode = get_node_number(target, nvar, 0)
    s1node = get_node_number(source1, nvar, 0)
    s2node = get_node_number(source2, nvar, 0)
    pt = g.predecessors(tnode)
    ps1path, s1path, s1pathnested = get_path_nodes_and_their_parents(g, s1node, tnode)
    ps2path, s2path, s2pathnested = get_path_nodes_and_their_parents(g, s2node, tnode)

    # print tnode, pt
    # print s1path, s2path

    # Generate w1
    if s1path:
        wset1 = exclude_intersection(pt, s1path)
        w1 = union([wset1, ps1path])
        print w1
    else:
        w1 = []

    # Generate w2
    if s2path:
        wset1 = exclude_intersection(pt, s2path)
        w2 = union([wset1, ps2path])
    else:
        w2 = []

    # Generate w
    if not s1path or not s2path:
        w = []
    else:
        wset1 = exclude_intersection(pt, union([s1path, s2path]))
        wset2 = exclude_intersection(ps1path, s2path)
        wset3 = exclude_intersection(ps2path, s1path)
        w = union([wset1, wset2, wset3])

    # Convert pt, ps1path, ps2path, s1path, s2path and w back to the set version
    pt = convert_nodes_to_listofset(pt, nvar)
    ps1path = convert_nodes_to_listofset(ps1path, nvar)
    ps2path = convert_nodes_to_listofset(ps2path, nvar)
    s1path = convert_nodes_to_listofset(s1path, nvar)
    s1pathnested = [convert_nodes_to_listofset(path, nvar) for path in s1pathnested]
    s2path = convert_nodes_to_listofset(s2path, nvar)
    s2pathnested = [convert_nodes_to_listofset(path, nvar) for path in s2pathnested]
    w1 = convert_nodes_to_listofset(w1, nvar)
    w2 = convert_nodes_to_listofset(w2, nvar)
    w = convert_nodes_to_listofset(w, nvar)

    if verbosity > 0:
        print '------- The causal path from the first source to the target is:'
        print s1pathnested
        print ''
        print '------- The causal path from the second source to the target is:'
        print s2pathnested
        print ''
        print '------- The condition of the first causal path includes:'
        print w1
        print '------- The condition of the second causal path includes:'
        print w2
        print '------- The MPID condition includes:'
        print w
        print ''

    return pt, s1pathnested, s2pathnested, ps1path, ps2path, w, w1, w2


# Help functions
def convert_nodes_to_listofset(nodes, nvar):
    '''Convert a list of nodes (in number) into a list of sets.'''
    return [get_node_set(node, nvar) for node in nodes]


def get_node_number(nodedict, nvar, gap):
    '''Convert a node in set version to the node number in the graph.'''
    return nodedict[0] + gap + nvar*abs(nodedict[1])


def get_node_set(nodenumber, nvar):
    '''Convert a node in number into a set (node index, time lag).'''
    # Get the variable index
    nodeindex = nodenumber % nvar

    # Get the time lag
    lag = nodenumber / nvar

    return (nodeindex, -lag)


def get_path_nodes_and_their_parents(g, s1node, tnode):
    '''Get the path from s1node to tnode and the parents of the path given a graph g.'''
    # Get the path from s1node to tnode
    pathall = nx.all_simple_paths(g, s1node, tnode)
    pathall = list(pathall)

    # Exclude the target node
    s1pathsnested = [p[:-1] for p in pathall]

    # Convert the paths to a list of nodes
    s1path = unique([node for path in s1pathsnested for node in path])

    # Get the parents of the paths
    # (1) get the parents of each node in all the paths
    parents = []
    for p in s1pathsnested:
        for node in p:
            parent = g.predecessors(node)
            parents += parent
    parents = unique(parents)
    # (2) exclude the parents which are also the nodes in the paths
    ps1path = exclude_intersection(parents, s1path)

    return ps1path, s1path, pathall


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


# def convert_causalDict_to_int(causalDict):
#     '''Convert the causalDict into a dictionary where the keys are integer (i.e., 0,1,2,3,4...)'''
#     var = causalDict.keys()
#     nvar = len(var)

#     # Create the mapping between the original var and the sequential integer
#     varmap = {}
#     for i in range(nvar):
#         varmap[var[i]] = i

#     # Convert the causalDict
#     causalDictInt = {}
#     for i in range(nvar):
#         causalDictInt[i] = []
#         for parent in causalDict[var[i]]:
#             causalDictInt.append([varmap(parent[0]), parent[1]])

#     return varmap, causalDictInt

if __name__ == '__main__':
    causalDict = {0: [(3, -1)],
                  1: [(3, -1)],
                  2: [(0, -1), (1, -1)],
                  3: []}
    verbosity = 1

    source1, source2 = (0, -1), (1, -1)
    target = (2, 0)

    search_mpid_condition(causalDict, source1, source2, target,
                          taumax=5, verbosity=verbosity)
