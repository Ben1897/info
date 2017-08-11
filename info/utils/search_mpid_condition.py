'''
This script is used for generating the condition of a MIP given:
(1) the causality relationship between different variables;
(2) the two sources and the target

Ref:
Jiang and Kumar PRE, in preparation (2017)

'''

import networkx as nx


def search_mpid_condition(causalDict, source1, source2, target, taumax=4, sidepath=False, verbosity=1):
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
    sidepath -- the decision of including the sidepath effect [bool]
    verbosity -- the printing level [int]

    Ouput:
    pt -- the parents of the target node [list of sets]
    s1path -- the causal path from the first source node to the target node [list of sets]
    s2path -- the causal path from the second source node to the target node [list of sets]
    s1pathnested -- the causal path from the first source node to the target node (nested) [list of list of sets]
    s2pathnested -- the causal path from the second source node to the target node (nested) [list of list of sets]
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
    g = nx.Graph()

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
            for parent_neighbor in causalDict[j]:
                isneighbor = is_neighbor(parent_neighbor)
                start = get_node_number(parent_neighbor, nvar, gap)
                # print get_node_set(start, 4), get_node_set(end, 4), isneighbor
                g.add_edge(start, end, start=start, end=end, isneighbor=isneighbor)

    # Generate pt, s1path, s2path
    tnode = get_node_number(target, nvar, 0)
    s1node = get_node_number(source1, nvar, 0)
    s2node = get_node_number(source2, nvar, 0)
    pt = find_parents_from_nodes(g, [tnode])
    s1pathnested, ps1path, s1path, ps1sidepath, s1sidepath = get_path_nodes_and_their_parents(g, s1node, tnode)
    s2pathnested, ps2path, s2path, ps2sidepath, s2sidepath = get_path_nodes_and_their_parents(g, s2node, tnode)

    # Generate w1
    if s1path and not sidepath:
        wset1 = exclude_intersection(pt, s1path)
        w1 = union([wset1, ps1path])
    elif s1path and sidepath:
        wset1 = exclude_intersection(pt, s1path)
        w1 = union([wset1, ps1path, ps1sidepath, s1sidepath])
    else:
        print "WARNING: the causal paths from the first source to the target is empty!"
        w1 = []

    # Generate w2
    if s2path:
        wset1 = exclude_intersection(pt, s2path)
        w2 = union([wset1, ps2path])
    elif s1path and sidepath:
        wset2 = exclude_intersection(pt, s2path)
        w2 = union([wset2, ps2path, ps2sidepath, s2sidepath])
    else:
        print "WARNING: the causal paths from the second source to the target is empty!"
        w2 = []

    # Generate w
    if not s1path or not s2path:
        print "WARNING: the causal paths from the two sources to the target is empty!"
        w = []
    elif not sidepath:
        wset1 = exclude_intersection(pt, union([s1path, s2path]))
        wset2 = exclude_intersection(ps1path, s2path)
        wset3 = exclude_intersection(ps2path, s1path)
        w = union([wset1, wset2, wset3])
    elif sidepath:
        wset1 = exclude_intersection(pt, union([s1path, s2path]))
        wset2 = exclude_intersection(ps1path, s2path)
        wset3 = exclude_intersection(ps2path, s1path)
        wset4 = exclude_intersection(union([s1sidepath, ps1sidepath]), union([s1path, s2path]))
        wset5 = exclude_intersection(union([s2sidepath, ps2sidepath]), union([s1path, s2path]))
        w = union([wset1, wset2, wset3, wset4, wset5])

    # Convert pt, ps1path, ps2path, s1path, s2path and w back to the set version
    pt = convert_nodes_to_listofset(pt, nvar)
    ps1path = convert_nodes_to_listofset(ps1path, nvar)
    ps2path = convert_nodes_to_listofset(ps2path, nvar)
    s1path = convert_nodes_to_listofset(s1path, nvar)
    s2path = convert_nodes_to_listofset(s2path, nvar)
    s1pathnested = [convert_nodes_to_listofset(path, nvar) for path in s1pathnested]
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

    return pt, s1path, s2path, s1pathnested, s2pathnested, w, w1, w2


# Help functions
def convert_nodes_to_listofset(nodes, nvar):
    '''Convert a list of nodes (in number) into a list of sets.'''
    return [get_node_set(node, nvar) for node in nodes]


def get_node_number(nodedict, nvar, gap):
    '''Convert a node in set version to the node number in the graph.'''
    return nodedict[0] + gap + nvar*abs(nodedict[1])


def is_neighbor(nodedict):
    '''Judge whether the node is the parent or neighbor of another node'''
    if nodedict[1] == 0:
        return True
    else:
        return False


def get_node_set(nodenumber, nvar):
    '''Convert a node in number into a set (node index, time lag).'''
    # Get the variable index
    nodeindex = nodenumber % nvar

    # Get the time lag
    lag = nodenumber / nvar

    return (nodeindex, -lag)


def get_path_nodes_and_their_parents(g, snode, tnode):
    '''Get the path from s1node to tnode and the parents of the path given a graph g.'''
    # Get the causal paths and the neighbors of the source node
    # which locate in the contemporaneous paths
    causalpaths, neighbors, causalpathsnested = find_causal_contemp_paths(g, snode, tnode)

    # Get the parents of the causal paths and the contemporaneous neighbors
    pcausalpaths = find_parents_from_nodes(g, causalpaths)
    pneighbors = find_parents_from_nodes(g, neighbors)

    return causalpathsnested, pcausalpaths, causalpaths, pneighbors, neighbors


def find_causal_contemp_paths(g, snode, tnode):
    '''Find the causal paths from the source node to the target node, and
       find its neighbors which locate in the contemporaneous paths.
       Return two lists of nodes:  causalpaths, neighbors'''
    # Get all the path from s1node to tnode
    pathall = nx.all_simple_paths(g, snode, tnode)
    pathall = list(pathall)

    # Distinguish the causal paths and contemporaneous paths from pathall
    causalpaths = []
    contemppaths, neighbors = [], []
    for p in pathall:
        iscausal, iscontemp = True, True
        for i in range(len(p)-1):
            start, end = p[i], p[i+1]
            edgeattr = g.get_edge_data(start, end)
            if edgeattr['isneighbor']:
                iscausal = False
            elif edgeattr['start'] != start:
                iscausal, iscontemp = False, False
                break
        if iscausal:
            causalpaths += [p]
        elif iscontemp:
            contemppaths += [p]
            neighbors += [p[1]]

    # Exclude the target node
    causalpaths_notarget = [p[:-1] for p in causalpaths]

    # Convert the paths to a list of nodes
    causalpaths_final = unique([node for path in causalpaths_notarget for node in path])

    return causalpaths_final, neighbors, causalpaths


def find_parents_from_nodes(g, nodes):
    """ Return the parents of one or more nodes from graph g."""
    # Get the parents of the paths
    # (1) get the parents of each node in all the paths
    parents = []
    for end in nodes:
        for start in g.neighbors(end):
            edgeattr = g.get_edge_data(start, end)
            if edgeattr['isneighbor'] or edgeattr['end'] != end:
                continue
            else:
                parents += [start]
    parents = unique(parents)
    # (2) exclude the parents which are also the nodes in the paths
    parents = exclude_intersection(parents, nodes)

    return parents


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
                  2: [(0, -1), (1, -1), (2, -1)],
                  3: []}
    verbosity = 1

    source1, source2 = (0, -1), (1, -1)
    target = (2, 0)

    search_mpid_condition(causalDict, source1, source2, target,
                          taumax=5, verbosity=verbosity)

    causalDict = {0: [(2, 0), (3, -1)],
                  1: [(3, -1)],
                  2: [(0, 0), (0, -1), (1, -1), (2, -1)],
                  3: []}
    verbosity = 1

    source1, source2 = (0, -1), (1, -1)
    target = (2, 0)

    search_mpid_condition(causalDict, source1, source2, target,
                          taumax=5, sidepath=True, verbosity=verbosity)
