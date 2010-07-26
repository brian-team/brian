'''
Code to separate Equations into independent subsets. This can be used in
STDP to allow for nonlinear equations to be given. Also, a modified version of
this code could probably be used to define independent equations for use in
compound solvers. For example, you could split into separate independent sets
and have one solver for each set. In some circumstances, this might be more
efficient (e.g. a linear and nonlinear component). It may need to be modified so
that the dependencies are directional, i.e. X depending on Y doesn't mean Y
depends on X (which you want in the case of the STDP separation), that way you
could have a linear solver for a subset and a nonlinear solver that depends on
some of the linear variables. In this case you would probably look for minimal
subgraphs which are allowed to have incoming edges but no outgoing edges,
construct solvers for them, and then construct a compound solver for the rest.
'''

if __name__ == '__main__':
    from brian.equations import Equations
    from brian.inspection import get_identifiers
else:
    from ..equations import Equations
    from ..inspection import get_identifiers
from collections import defaultdict
from copy import copy

__all__ = ['separate_equations']

def next_independent_subgraph(G):
    '''
    Returns an independent subgraph of G and removes that subgraph from G.
    
    Here G is a graph represented as a dictionary, so that G[node] is a set
    of all the neighbours of node. The caller is responsible for constructing
    a well formed G which has all the appropriate nodes, and where edges in
    the graph are both ways so that b in G[a] <=> a in G[b].
    '''
    # We start from just one (randomly selected) node and iteratively add
    # elements to the subgraph from the set of neighbours of the subgraph
    # until there are no more neighbours to be added, and then we return.
    # This algorithm may not be efficient but we're likely only dealing with
    # fairly small graphs anyway.
    nodes = dict([G.popitem()])
    found_new = True
    while found_new:
        found_new = False
        for node, targets in nodes.items():
            for target in targets:
                if target in G: # if target not in G, we assume it has already been added
                    nodes[target] = G.pop(target)
                    found_new = True
                else: # this should be true if the caller has constructed G correctly
                    assert target in nodes
    return nodes

def separate_equations(eqs, additional_dependencies=[]):
    eqs.prepare()
    all_vars = eqs._string.keys()
    # Construct a list of dependency sets, where each variable in each
    # dependency set induces a dependency on each other variable in each
    # dependency set
    depsets = []
    for var in all_vars:
        ids = set(get_identifiers(eqs._string[var]))
        ids = ids.intersection(all_vars)
        ids.add(var)
        depsets.append(ids)
    for expr in additional_dependencies:
        ids = set(get_identifiers(expr))
        ids = ids.intersection(all_vars)
        depsets.append(ids)
    # Construct a graph deps which indicates what variable depends on which
    # other variables (or is depended on by other variables).
    deps = defaultdict(set)
    for ids in depsets:
        for id1 in ids:
            for id2 in ids:
                deps[id1].add(id2)
    # Extract all the independent subgraphs
    ind_graphs = []
    while len(deps):
        ind_graphs.append(set(next_independent_subgraph(deps).keys()))
    if len(ind_graphs) == 1:
        return [eqs]
    # Finally, we construct an Equations object for each of the subgraphs
    ind_eqs = []
    for G in ind_graphs:
        neweqs = Equations()
        for var in G:
            if var in eqs._eq_names:
                neweqs.add_eq(var, eqs._string[var], eqs._units[var],
                              local_namespace=eqs._namespace[var])
            elif var in eqs._diffeq_names:
                nonzero = var in eqs._diffeq_names_nonzero
                neweqs.add_diffeq(var, eqs._string[var], eqs._units[var],
                                  local_namespace=eqs._namespace[var],
                                  nonzero=nonzero)
            elif var in eqs._alias.keys():
                neweqs.add_alias(var, eqs._string[var].strip())
            else:
                assert False
        ind_eqs.append(neweqs)
    return ind_eqs

if __name__ == '__main__':
    from brian import *
#    T = second
#    eqs = Equations('''
#    da/dt = y/T : 1
#    db/dt = x/T : 1
#    c : 1
#    x = c
#    y = x*x : 1
#    dd/dt = e/T : 1
#    de/dt = d/T : 1
#    df/dt = e/T : 1
#    ''')

    tau_pre = tau_post = 10 * ms
    eqs = Equations("""
    dA_pre/dt  = -A_pre/tau_pre   : 1
    dA_post/dt = -A_post/tau_post : 1
    """)

    eqs_sep = separate_equations(eqs)
    for e in eqs_sep:
        print '******************'
        print e
