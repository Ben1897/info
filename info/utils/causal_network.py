'''
This script is used for creating a causal network, which is able to:
(1) find the parent(s) and neighbor(s) of a node
(2) check whether two nodes are linked with a directed link or an undirected link
(3) find the causal path between two nodes
(4) find the conditions for calculating the momentary information transfer (MIT) between two nodes
(5) find the conditions for calculating the momentary partial information decomposition (MPID) between two sources and a target

Ref:
Runge PRE (2015)
Jiang and Kumar PRE, under review (2017)

class causal_network()
    __init__()
    search_parents_neighbors()
    search_paths()
    check_links()
    search_mit_condition()
    search_mpid_condition()

convert_nodes_to_listofset()
get_node_number()
is_neighbor()
get_node_set()
get_path_nodes_and_their_parents()
get_causal_contemp_paths()
get_parents_from_nodes()
get_neighbors_from_nodes()
unique()
intersect()
union()
exclude_intersection()

'''

import networkx as nx


class causal_network(object):
    """A class for causal network."""

    def __init__(self, causalDict, taumax=6):
        """
        Input:
        causalDict -- dictionary of causal relationships, where
                      the keys are the variable at t time [int]
                      the values are the parents of the corresponding node [list of sets].
                      e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
        taumax -- the maximum time lag [int]

        """
        self.causalDict = causalDict
        self.taumax     = taumax

        # Create an empty directed graph
        g = nx.DiGraph()

        # Assign all the nodes into the directed graph
        var    = causalDict.keys()
        nvar   = len(var)
        nnodes = nvar*(taumax+1)
        g.add_nodes_from(range(nnodes))

        # Assign all the edges
        for i in range(taumax+1):
            gap           = nvar*i
            involed_nodes = range(gap, gap+nvar)
            for j in range(nvar):
                end = involed_nodes[j]
                for parent_neighbor in causalDict[j]:
                    isneighbor = is_neighbor(parent_neighbor)
                    start      = get_node_number(parent_neighbor, nvar, gap)
                    if isneighbor:  # Here an undirected edge is miciced by two directed edges with the opposite directions
                        g.add_edge(start, end, isneighbor=isneighbor)
                        g.add_edge(end, start, isneighbor=isneighbor)
                    else:
                        g.add_edge(start, end, isneighbor=isneighbor)

        self.var    = var
        self.nvar   = nvar
        self.nnodes = g.number_of_nodes()
        self.g      = g

    def __check_node(self, target):
        """
        Check whether the target node is in the valid graph by fulfilling the following conditions:
            (1) var_index is within self.var
            (2) lag is smaller than and equal to taumax
        Input:
        target -- the target node [set (var_index, lag)]

        """
        if target[0] not in self.var:
            raise Exception("The target variable %d is not in the valid variable set!" % target[0])

        if target[1] > self.taumax:
            raise Exception("The time step of the target node %d is larger than the taumax!" % target[1])

    def search_parents_neighbors(self, target, verbosity=1):
        """
        Find the parent(s) and neighbor(s) of a node.

        Input:
        target -- the target node [set (var_index, lag)]
        Output:
        {'parents': list of set, 'neighbors': list of set}

        """
        g, nvar = self.g, self.nvar

        # Check whether the node is in the causal network
        self.__check_node(target)

        # Get the node number
        tnode = get_node_number(target, nvar, 0)

        # Get its parents
        parents = get_parents_from_nodes(g, [tnode])

        # Get its neighbors
        neighbors = get_neighbors_from_nodes(g, [tnode])

        if not parents and not neighbors and verbosity==1:
            print 'No parents and neighbors for the node:'
            print target
            return {'parents': [], 'neighbors': []}
        else:
            return {'parents': convert_nodes_to_listofset(parents, nvar),
                    'neighbors': convert_nodes_to_listofset(neighbors, nvar)}

    def search_paths(self, source, target, nested=False, verbosity=1):
        """
        Find the causal paths and contemporatneous sidepaths between a source and a target.

        Input:
        source -- the source node [set (var_index, lag)]
        target -- the target node [set (var_index, lag)]
        Output:
        if nested     -- {'causal': list of paths (each of which is a list of set),
                          'contemp': list of paths (each of which is a list of set)}
        if not nested -- {'causal': list of set, 'contemp': list of set}

        """
        g, nvar = self.g, self.nvar

        # Check whether the node is in the causal network
        self.__check_node(source)
        self.__check_node(target)

        # Get the node number
        snode = get_node_number(source, nvar, 0)
        tnode = get_node_number(target, nvar, 0)

        # Get the caual paths
        causalpaths, causalpaths_nest, contemppaths, contemppaths_nest, _ = get_causal_contemp_paths(g, snode, tnode)

        # Return
        if nested:
            return {'causal': [convert_nodes_to_listofset(path, nvar) for path in causalpaths_nest],
                    'contemp': [convert_nodes_to_listofset(path, nvar) for path in contemppaths_nest]}
        else:
            return {'causal': convert_nodes_to_listofset(causalpaths, nvar),
                    'contemp': convert_nodes_to_listofset(contemppaths, nvar)}

    def check_links(self, source, target, verbosity=1):
        """
        Check whether two nodes are linked with a directed link, an undirected link, a causal path or a contemporaneous sidepath.

        Input:
        source -- the source node [set (var_index, lag)]
        target -- the target node [set (var_index, lag)]
        Output:
        directed link            -- 'directed'
        contemporaneous link     -- 'undirected'
        causal path              -- 'causal'
        contemporaneous sidepath -- 'contemp'
        none of the above        -- None

        """
        # Check whether the link is a directed link or an undirected link
        parents_neighors = self.search_parents_neighbors(target, verbosity=verbosity)
        if source in parents_neighors['parents']:
            return 'directed'
        elif source in parents_neighors['neighbors']:
            return 'contemporaneous'

        # Check whether the link is a causal path or a contemporaneous sidepath
        paths = self.search_paths(source, target, nested=True, verbosity=verbosity)
        if paths['causal']:
            causal_paths = [node for path in paths['causal'] for node in path]
            if source in causal_paths:
                return 'causalpath'
        elif paths['contemp']:
            contemp_paths = [node for path in paths['contemp'] for node in path]
            if source in contemp_paths:
                return 'contempsidepath'

        # If none of them is found, return None
        if verbosity==1:
            print '%s and %s are not linked through the following types:' % (source, target)
            print 'directed link, contemporaneous link, causal path and contemporaneous sidepath!'
        return None

    def search_mit_condition(self, source, target, sidepath=False, verbosity=1):
        """
        Find the conditions for calculating the momentary information transfer (MIT) between between a source and a target
        based on the link type:
            'directed'        -- the condition for MIT
            'causalpath'      -- the condition for MITP (with or without sidepath effect based on the input sidepath)
            'contemporaneous' -- the condition for MIT for two contemporaneous nodes
            'contempsidepath' -- no condition

        Input:
        source -- the source node [set (var_index, lag)]
        target -- the target node [set (var_index, lag)]
        sidepath -- whether including the contemporaneous sidepaths [bool]
        Output:
        the condition for MIT [list of sets]

        """
        g, nvar = self.g, self.nvar

        # Check whether the node is in the causal network
        self.__check_node(source)
        self.__check_node(target)

        # Get the node number
        snode = get_node_number(source, nvar, 0)
        tnode = get_node_number(target, nvar, 0)

        # Get the parents of the target node
        pt = get_parents_from_nodes(g, [tnode])

        linktype = self.check_links(source, target, verbosity=verbosity)

        # Get the condition for MIT
        if linktype == 'contemporaneous':                                     # linked by a contemporaneous undirected link
            w1 = get_parents_from_nodes(g, [snode])
            w2 = get_parents_from_nodes(g, [tnode])
            w3 = exclude_intersection(get_neighbors_from_nodes(g, [snode]), [tnode])
            w4 = exclude_intersection(get_neighbors_from_nodes(g, [tnode]), [snode])
            w5 = get_parents_from_nodes(w3)
            w6 = get_parents_from_nodes(w4)
            w  = union([w1, w2, w3, w4, w5, w6])

        else:                                              # linked by a directed link
            # w = union([pcpath, exclude_intersection(pt, cpath)])
            w1 = get_parents_from_nodes(g, [snode])
            w2 = exclude_intersection(get_parents_from_nodes(g, [tnode]), [snode])
            w = union([w1, w2])

        w = convert_nodes_to_listofset(w, nvar)

        if verbosity:
            print "The link type between %s and %s is %s" % (source, target, linktype)
            if linktype == 'causalpath' and verbosity > 1:
                print "The path from %s to %s is ---" % (source, target)
                print [convert_nodes_to_listofset(path, nvar) for path in cpathnested]
            print "The number of conditions from %s to %s is %d, including:" % (source, target, len(w))
            print w

        return w

    def search_mitp_condition(self, source, target, sidepath=False, verbosity=1):
        """
        Find the conditions for calculating the momentary information transfer (MIT) between between a source and a target
        based on the link type:
            'directed'        -- the condition for MIT
            'causalpath'      -- the condition for MITP (with or without sidepath effect based on the input sidepath)
            'contemporaneous' -- the condition for MIT for two contemporaneous nodes
            'contempsidepath' -- no condition

        Input:
        source -- the source node [set (var_index, lag)]
        target -- the target node [set (var_index, lag)]
        sidepath -- whether including the contemporaneous sidepaths [bool]
        Output:
        the condition for MITP [list of sets]

        """
        g, nvar = self.g, self.nvar

        # Check whether the node is in the causal network
        self.__check_node(source)
        self.__check_node(target)

        # Get the node number
        snode = get_node_number(source, nvar, 0)
        tnode = get_node_number(target, nvar, 0)

        # Get the parents of the target node
        pt = get_parents_from_nodes(g, [tnode])

        linktype = self.check_links(source, target, verbosity=verbosity)

        # Get the condition for MITP
        if linktype not in ['causalpath', 'directed']:                                              # linked by a directed link
            w = []
            if verbosity == 1:
                print "The two nodes %s and %s are not connected by a causal path!" % (source, target)

        else:                                              # linked by a causal path
            # Get the causal path, the parent(s) of the causal path, the neighbor(s) in the contemporaneous sidepath(s)
            # and the parents of the neighbor(s) of the source
            cpathnested, pcpath, cpath, psidepathneighbor, sidepathneighbor = get_path_nodes_and_their_parents(g, snode, tnode)
            if sidepath:   # with sidepath
                w1 = union([pcpath, exclude_intersection(pt, cpath)])
                w2 = union([sidepathneighbor, psidepathneighbor])
                w  = union([w1, w2])
            else:          # without sidepath
                w = union([pcpath, exclude_intersection(pt, cpath)])

        w = convert_nodes_to_listofset(w, nvar)

        if verbosity:
            print "The link type between %s and %s is %s" % (source, target, linktype)
            if linktype == 'causalpath' and verbosity > 1:
                print "The path from %s to %s is ---" % (source, target)
                print [convert_nodes_to_listofset(path, nvar) for path in cpathnested]
            print "The number of conditions from %s to %s is %d, including:" % (source, target, len(w))
            print w

        return w

    def search_mpid_condition(self, source1, source2, target, sidepath=False, verbosity=1):
        """
        Find the conditions for calculating the momentary partial information decomposition (MPID) between two sources and a target.

        Input:
        source1  -- the first source node [set (var_index, lag)]
        source2  -- the second source node [set (var_index, lag)]
        target   -- the target node [set (var_index, lag)]
        sidepath --  whether including the contemporaneous sidepaths [bool]
        Output:
        the condition for MPID [list of sets]

        """
        g, nvar = self.g, self.nvar

        # Check whether the node is in the causal network
        self.__check_node(source1)
        self.__check_node(source2)
        self.__check_node(target)

        # Check whether the two sources are linked with the target through causal paths
        linktype1 = self.check_links(source1, target, verbosity=verbosity)
        linktype2 = self.check_links(source1, target, verbosity=verbosity)
        if linktype1 not in ['causalpath', 'directed']:
            if verbosity == 1:
                print "The source %s and the target %s are not linked by a causal path" % (source1, target)
            return []
        if linktype2 not in ['causalpath', 'directed']:
            if verbosity == 1:
                print "The source %s and the target %s are not linked by a causal path" % (source2, target)
            return []

        # Get the node number
        s1node = get_node_number(source1, nvar, 0)
        s2node = get_node_number(source2, nvar, 0)
        tnode  = get_node_number(target, nvar, 0)

        # Get the parents of the target node
        pt = get_parents_from_nodes(g, [tnode])

        # Get the causal path, the parent(s) of the causal path, the neighbor(s) in the contemporaneous sidepath(s)
        # and the parents of the neighbor(s) of the two sources
        cpathnested1, pcpath1, cpath1, psidepathneighbor1, sidepathneighbor1 = get_path_nodes_and_their_parents(g, s1node, tnode)
        cpathnested2, pcpath2, cpath2, psidepathneighbor2, sidepathneighbor2 = get_path_nodes_and_their_parents(g, s2node, tnode)

        # Get the conditions for MPID
        w1 = exclude_intersection(pt, union([cpath1, cpath2]))
        w2 = exclude_intersection(pcpath1, cpath2)
        w3 = exclude_intersection(pcpath2, cpath1)
        if sidepath:  # with sidepath
            w4 = exclude_intersection(union([sidepathneighbor1, psidepathneighbor1]), union([cpath1, cpath2]))
            w5 = exclude_intersection(union([sidepathneighbor2, psidepathneighbor2]), union([cpath1, cpath2]))
            w = union([w1, w2, w3, w4, w5])
        else:         # without sidepath
            w = union([w1, w2, w3])

        w = convert_nodes_to_listofset(w, nvar)

        if verbosity:
            print "The number of conditions from %s and %s to %s is %d, including:" % (source1, source2, target, len(w))
            print w

        return w


# Help functions
def convert_nodes_to_listofset(nodes, nvar):
    '''Convert a list of nodes (in number) into a list of sets.'''
    return [get_node_set(node, nvar) for node in nodes]


def get_node_number(nodedict, nvar, lag):
    '''Convert a node in set version to the node number in the graph.'''
    return nodedict[0] + lag + nvar*abs(nodedict[1])


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
    '''Get the path from snode to tnode and the parents of the path given a graph g.'''
    # Get the causal paths and the neighbors of the source node
    # which locate in the contemporaneous paths
    causalpaths, causalpathsnested, _, _, neighbors = get_causal_contemp_paths(g, snode, tnode)

    # Get the parents of the causal paths and the contemporaneous neighbors
    pcausalpaths = get_parents_from_nodes(g, causalpaths)
    pneighbors = get_parents_from_nodes(g, neighbors)
    # print pneighbors
    # print pcausalpaths

    return causalpathsnested, pcausalpaths, causalpaths, pneighbors, neighbors


def get_causal_contemp_paths(g, snode, tnode):
    '''Find the causal paths from the source node to the target node, and
       find its neighbors which locate in the contemporaneous paths.

       Return five lists of nodes:
            causalpaths_final, causalpaths, contemppaths_final, contemppaths, neighbors
    '''
    # Get all the path from s1node to tnode
    pathall = nx.all_simple_paths(g, snode, tnode)
    pathall = list(pathall)

    # Distinguish the causal paths and contemporaneous paths from pathall
    causalpaths = []
    contemppaths, neighbors = [], []
    for p in pathall:
        iscausal, iscontemp = True, False
        for i in range(len(p)-1):
            start, end = p[i], p[i+1]
            edgeattr = g[start][end]
            if edgeattr['isneighbor'] and i == 0:
                iscausal, iscontemp = False, True
                break
            elif edgeattr['isneighbor'] and i != 0 and not iscontemp:
                iscausal = False
        if iscausal:
            causalpaths += [p]
        elif iscontemp:
            contemppaths += [p]
            neighbors += [p[1]]

    # Exclude the target node
    causalpaths_notarget = [p[:-1] for p in causalpaths]
    contemppaths_notarget = [p[:-1] for p in contemppaths]

    # Convert the paths to a list of nodes
    causalpaths_final = unique([node for path in causalpaths_notarget for node in path])
    contemppaths_final = unique([node for path in contemppaths_notarget for node in path])

    return causalpaths_final, causalpaths, contemppaths_final, contemppaths, neighbors


def get_parents_from_nodes(g, nodes):
    """ Return the parents of one or more nodes from graph g."""
    # Get the parents of the nodes
    # (1) get the parents of each node in all the nodes
    parents = []
    for end in nodes:
        for start in g.predecessors(end):
            edgeattr = g.get_edge_data(start, end)
            if not edgeattr['isneighbor']:
                parents += [start]
    parents = unique(parents)
    # (2) exclude the parents which are also the nodes in the nodes
    parents = exclude_intersection(parents, nodes)

    return parents


def get_neighbors_from_nodes(g, nodes):
    """ Return the neighbors of one or more nodes from graph g."""
    # Get the neighbors of the nodes
    # (1) get the neighbors of each node in all the nodes
    neighbors = []
    for end in nodes:
        for start in g.predecessors(end):
            edgeattr = g.get_edge_data(start, end)
            if edgeattr['isneighbor']:
                neighbors += [start]
    neighbors = unique(neighbors)
    # (2) exclude the neighbors which are also the nodes in the nodes
    neighbors = exclude_intersection(neighbors, nodes)

    return neighbors


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


# Unused function
def search_mpid_condition(causalDict, source1, source2, target, threeconditions=False, taumax=4, sidepath=False, verbosity=1):
    '''
    Generate the condition of a MIP.

    Input:
    causalDict -- dictionary of causal relationships, where
                  the keys are the variable at t time [int]
                  the values are the parents of the corresponding node [list of sets].
                  e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
    source1 -- the first source node [set]
    source2 -- the second source node [set]
    threeconditions -- the decision of calculating the conditions for the two causal paths (i.e., w1 and w2) [bool]
    target -- the target node [set]
    taumax -- the maximum time lag [int]
    sidepath -- the decision of including the sidepath effect [bool]
    verbosity -- the printing level [int]

    Output:
    pt -- the parents of the target node [list of sets]
    s1path -- the path from the first source node to the target node [list of sets]
    s2path -- the path from the second source node to the target node [list of sets]
    s1pathnested -- the causal path from the first source node to the target node (nested) [list of list of sets]
    s2pathnested -- the causal path from the second source node to the target node (nested) [list of list of sets]
    ps1path -- the parents of s1path [list of sets]
    ps2path -- the parents of s2path [list of sets]
    w -- the condition of the two causal paths [list of sets]
    w1 -- the condition of the first causal path empty if threeconditions is False [list of sets]
    w2 -- the condition of the second causal path empty if threeconditions is False [list of sets]
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
            for parent_neighbor in causalDict[j]:
                isneighbor = is_neighbor(parent_neighbor)
                start = get_node_number(parent_neighbor, nvar, gap)
                if isneighbor:  # Here an undirected edge is miciced by two directed edges with the opposite directions
                    g.add_edge(start, end, isneighbor=isneighbor)
                    g.add_edge(end, start, isneighbor=isneighbor)
                else:
                    g.add_edge(start, end, isneighbor=isneighbor)

    # Generate pt, s1path, s2path
    tnode = get_node_number(target, nvar, 0)
    s1node = get_node_number(source1, nvar, 0)
    s2node = get_node_number(source2, nvar, 0)
    # pt = g.predecessors(tnode)
    pt = get_parents_from_nodes(g, [tnode])
    s1pathnested, ps1path, s1path, ps1sidepathneighbor, s1sidepathneighbor = get_path_nodes_and_their_parents(g, s1node, tnode)
    s2pathnested, ps2path, s2path, ps2sidepathneighbor, s2sidepathneighbor = get_path_nodes_and_their_parents(g, s2node, tnode)

    # Generate w1
    if s1path and threeconditions:
        wset1 = exclude_intersection(pt, s1path)
        w1 = union([wset1, ps1path])
    else:
        w1 = []

    # Generate w2
    if s2path and threeconditions:
        wset1 = exclude_intersection(pt, s2path)
        w2 = union([wset1, ps2path])
    else:
        w2 = []

    # Generate w
    if not s1path or not s2path:
        print "WARNING: the causal paths from the two sources to the target is empty!"
        w = []
    elif not sidepath:
        print "No sidepath is assumed."
        wset1 = exclude_intersection(pt, union([s1path, s2path]))
        wset2 = exclude_intersection(ps1path, s2path)
        wset3 = exclude_intersection(ps2path, s1path)
        w = union([wset1, wset2, wset3])
    elif sidepath:
        print "Sidepath is assumed."
        wset1 = exclude_intersection(pt, union([s1path, s2path]))
        wset2 = exclude_intersection(ps1path, s2path)
        wset3 = exclude_intersection(ps2path, s1path)
        wset4 = exclude_intersection(union([s1sidepathneighbor, ps1sidepathneighbor]), union([s1path, s2path]))
        wset5 = exclude_intersection(union([s2sidepathneighbor, ps2sidepathneighbor]), union([s1path, s2path]))
        w = union([wset1, wset2, wset3, wset4, wset5])

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
        if threeconditions:
            print '------- The condition of the first causal path includes:'
            print w1
            print '------- The condition of the second causal path includes:'
            print w2
        print '------- The MPID condition includes:'
        print w
        print ''

    return pt, s1path, s2path, s1pathnested, s2pathnested, w, w1, w2

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
    causalDict = {0: [(1, 0), (3, -1)],
                  # 1: [(3, -1)],
                  1: [(2, -1)],
                  2: [(0, -1), (1, -1), (3, -1)],
                  3: [(3, -1)]}
    verbosity = 1

    source1, source2 = (0, -1), (3, -1)
    target = (2, 0)

    search_mpid_condition(causalDict, source1, source2, target, threeconditions=True,
                          taumax=5, sidepath=True, verbosity=verbosity)
