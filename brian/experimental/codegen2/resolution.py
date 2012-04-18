from brian import *
from symbols import *
from blocks import *
from codeitems import *
from statements import *
from formatting import *
from languages import *
from dependencies import *

__all__ = ['resolve']

def resolve(item, symbols, namespace=None):
    '''
    Resolves ``symbols`` in ``item`` in the optimal order.
    
    The first stage of this algorithm is to construct a dependency graph
    on the symbols.
    
    The optimal order is resolve loops as late as possible.
    We actually construct the inverse of the resolution order, which is the
    intuitive order (i.e. if the first thing we do is loop over a variable, then
    that variable is the last symbol we resolve).
    
    We start by
    finding the set of symbols which have no dependencies. The graph is
    acyclic so this always possible. Then, among those candidates, if possible
    we choose loopless symbols first (this corresponds to executing loops as
    late as possible). With this symbol removed from the graph we repeat until
    all symbols are placed in order.
    
    We then resolve in reverse order (because we start with the inner loop
    and add code outwards). At the beginning of this stage, vectorisable is
    set to ``True``. But after we encounter the first multi-valued symbol
    we set ``vectorisable`` to ``False`` (we can only vectorise one loop, and
    it has to be the innermost one). This vectorisation is used by both Python
    and GPU but not C++. Each resolution step calls :meth:`CodeItem.resolve` on
    the output of the previous stage.
    '''
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
    vectorisable = True
    for var in resolution_order:
        sym = symbols[var]
        read = Read(var) in item.dependencies
        write = Write(var) in item.dependencies
        log_msg = 'Resolving '+var+' '
        if read:
            log_msg += '(read) '
        if write:
            log_msg += '(write) '
        if vectorisable:
            log_msg += '(vectorisable) '
        log_msg += 'dependencies '+str(item.dependencies)
        log_info('brian.codegen2.resolve', log_msg)
        item = sym.resolve(read, write, vectorisable, item, namespace)
        if sym.multi_valued():
            vectorisable = False
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
