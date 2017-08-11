"""
This file is used for generating the time series graph and the process graph of a causality network
by using graphviz.

Author: Peishi Jiang
email: shixijps@gmail.com

create_ts_graph()
get_node_name()

"""
# Load the modules
from graphviz import Digraph


def create_ts_graph(causalDict, varnames, lagmax=4,
                    highlightednodes=[], highlightededges=[], engine='neato'):
    '''Generate a time series graph.
    Inputs:
    causalDict -- dictionary of causal relationships, where
                  the keys are the variable at t time [int]
                  the values are the parents of the corresponding node [list of sets].
                  e.g., {0: [(0,-1), (1,-1)], 1: [(0,-2), (1,-1)]}
    varnames -- a list of variable names [list]
    lagmax -- the maximum lag for plotting
    highlightednodes -- a list of nodes to be highlighted [list of set]
    highlightededges -- a list of paths, each of which contains a sequence of directed nodes [list of list of [set]]
    Output:
    g -- the graph
    '''
    g = Digraph('G', engine=engine)
    g.attr(rankdir='LR')

    var = causalDict.keys()
    nvar = len(var)

    # Check whether the number of variables is correct
    if nvar != len(varnames):
        raise Exception('The number of variable names does not complies with the number of keys in causalDict!')

    # Generate a subgraph for each variable,
    # and generate all the required nodes for the subgraph
    for i in range(nvar):
        varname = varnames[i]
        with g.subgraph(name=varname) as c:
    #         c.attr(constraint='false')
            for j in range(lagmax+1):
                nodename = get_node_name(varname, j)
                # Check whether the node needs to be highlighted
                if (i,-j) in highlightednodes:
                    c.node(nodename, fixedsize='true', label=nodename,
                           pos=str((lagmax+1-j)*1.2)+','+str(nvar-i)+'!', style='filled')
                else:
                    c.node(nodename, fixedsize='true', label=nodename,
                           pos=str((lagmax+1-j)*1.2)+','+str(nvar-i)+'!')

    # Convert the highlighted edges into a list
    hiedges = []
    if highlightededges:
        for paths in highlightededges:
            for p in paths:
                for i in range(len(p)-1):
                    start_var, start_lag = varnames[p[i][0]], abs(p[i][1])
                    end_var, end_lag = varnames[p[i+1][0]], abs(p[i+1][1])
                    start = get_node_name(start_var, start_lag)
                    end = get_node_name(end_var, end_lag)
                    edge = (start, end)
                    hiedges.append(edge)

    # Connecting nodes
    if hiedges:
        g.edge_attr.update(color="grey")
    for i in range(nvar):
        var1 = varnames[i]
        for j in range(lagmax):
            if j == 0:
                end = var1 + '(t)'
            else:
                end = var1 + '(t-' + str(j) + ')'
            for k in causalDict[i]:
                var2 = varnames[k[0]]
                lag = j+abs(k[1])
                if lag == 0:
                    start = var2 + '(t)'
                else:
                    start = var2 + '(t-' + str(lag) + ')'
                # Connect start and end
                if (start, end) in hiedges:
                    g.edge(start, end, color="black")
                else:
                    g.edge(start, end)

    return g


# Help functions
def get_node_name(var, i):
    '''Get the node name.'''
    if i == 0:
        nodename = var + '(t)'
    else:
        nodename = var + '(t-' + str(i) + ')'
    return nodename


if __name__ == '__main__':
    from info.utils.search_mip_condition import search_mip_condition

    savedir = '../../test/figures'
    causalDict = {0: [(1,-1), (2,-1)],
                  1: [(0,-1), (1,-1), (2,-1)],
                  2: [(0,-1), (1,-1), (2,-1)]}
    lagmax = 3
    varnames = ['X1', 'X2', 'X3']
    engine='neato'
    ft = 'png'
    filename='test'
    source1, source2, target = (1,-2), (2,-1), (1,0)

    pt, s1path, s2path, ps1path, ps2path, w = search_mip_condition(causalDict, source1, source2, target)

    highlightednodes = [source1, source2, target]
    highlightededges = [s1path, s2path]

    g = create_ts_graph(causalDict, varnames, lagmax=lagmax,
                highlightednodes=highlightednodes, highlightededges=highlightededges, engine='neato')
    g.pipe(format=ft)
    g.view(filename=filename, directory=savedir)
