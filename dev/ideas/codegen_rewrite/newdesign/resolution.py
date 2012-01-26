from brian import *
from symbols import *
from blocks import *
from codeitems import *
from statements import *
from formatting import *
from languages import *

__all__ = ['resolve']

def resolve(item, symbols, namespace=None):
    if namespace is None:
        namespace = {}
    # stage 1, build the dependency graph
    depgraph = {}
    toprocess = get_read_or_write_dependencies(item.dependencies)
    while len(toprocess):
        var = toprocess.pop()
        if var not in symbols:
            # what should we do in this situation? hope that it will work
            # because the var is actually resolved or is a symbol like exp
            # that will be in the compiler and python namespaces?
            continue
        if var in depgraph:
            continue
        sym = symbols[var]
        deps = get_read_or_write_dependencies(sym.dependencies())
        # we are only interested in resolvable symbols, i.e. those in the
        # symbols dict, this should be all of them but possibly some might
        # not be (e.g. exp, as above)
        deps = [dep for dep in deps if dep in symbols]
        depgraph[var] = set(deps)
        toprocess.update(deps)
    # stage 2, compute the resolution order
    resolution_order = []
    while(len(depgraph)):
        # the candidates for resolution are those with no outgoing edges (and
        # since the graph is directed and acyclic there must be one)
        candidates = set(var for var, edges in depgraph.iteritems() if len(edges)==0)
        # just in case the graph isn't DA, raise an error
        if len(candidates)==0:
            raise ValueError("Graph is not acyclic!")
        # optimisation: try to resolve symbols without loops first
        loopless_candidates = set(var for var in candidates if not symbols[var].resolution_requires_loop())
        if len(loopless_candidates):
            candidates = loopless_candidates
        var = candidates.pop()
        for node, edges in depgraph.iteritems():
            edges.discard(var)
        del depgraph[var]
        resolution_order.append(var)
    resolution_order = resolution_order[::-1]
    # stage 3, do the resolving
    for var in resolution_order:
        sym = symbols[var]
        read = Read(var) in item.dependencies
        write = Write(var) in item.dependencies
        print 'Resolving', var,
        if read:
            print '(read)',
        if write:
            print '(write)',
        print 'dependencies', item.dependencies
        item = sym.resolve(read, write, item, namespace)
    return item, namespace
    
if __name__=='__main__':
    item = MathematicalStatement('x', ':=', 'y*y')
    y = zeros(10)
    #idx = array([1, 3, 5])
    subset = False
    #language = PythonLanguage()
    language = CLanguage()
    sym_y = ArraySymbol(y, 'y', language,
                      index='idx',
                      subset=subset,
                      )
    if subset:
        N = 'idx_arr_len'
        sym_idx = IndexSymbol('idx', N, language, index_array='idx_arr')
    else:
        N = 'y_len'
        sym_idx = IndexSymbol('idx', N, language)
    symbols = {'y':sym_y, 'idx':sym_idx}
    item, namespace = resolve(item, symbols)
    
    print 'Code:\n', indent_string(item.convert_to(language, symbols)),
    print 'Dependencies:', item.dependencies
    print 'Resolved:', item.resolved
    print
    print 'Namespace:', namespace.keys()
