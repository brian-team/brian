from brian import *
from symbols import *
from blocks import *
from statements import *
from dependencies import *

__all__ = ['CodeGenConnection',
           'DenseMatrixSymbols',
           'SparseMatrixSymbols',
           ]

def get_connection_variable(C, modulation=False):
    if modulation:
        G = C.source
        nstate = C._nstate_mod
    else:
        G = C.target
        nstate = C.nstate
    varname = str(nstate)
    for k, v in G.var_index.iteritems():
        if v==nstate and isinstance(k, str):
            varname = k
    return varname

class CodeGenConnection(Connection):
    def __init__(self, *args, **kwds):
        self._language = kwds.pop('language')
        Connection.__init__(self, *args, **kwds)
    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if not hasattr(self, '_use_codegen'):
            if not self.W.__class__ in [SparseConnectionMatrix,
                                        DenseConnectionMatrix,
                                        ]:
                log_warn('brian.codegen', "Only sparse and dense matrix supported.")
                self._use_codegen = False
        if not hasattr(self, '_use_codegen'):
            # allow ourselves the option of disabling codegen for cases where
            # it doesn't yet work (but we ignore this for the moment)
            self._use_codegen = True
            # preparation
            language = self._language
            targetvar = get_connection_variable(self)
            if self._nstate_mod is not None:
                modulation = True
                modvar = get_connection_variable(self, modulation=True)
                connection_code = '{targetvar} += _w*_sourcevar_{modvar}'.format(
                    targetvar=targetvar, modvar=modvar)
            else:
                modulation = False
                connection_code = '{targetvar} += _w'.format(targetvar=targetvar)
            statements = statements_from_codestring(connection_code,
                                                    defined=set(['w']),
                                                    eqs=self.target._eqs,
                                                    infer_definitions=True)      
            symbols = get_neuron_group_symbols(self.target, language,
                                               index='_target_index')
            if modulation:
                src_symbols = get_neuron_group_symbols(self.source, language,
                                               index='_source_index',
                                               prefix='_sourcevar_')
                symbols.update(src_symbols)
            symbols['_source_index'] = ArrayIndex('_source_index', '_spikes',
                                                  language,
                                                  array_len='_numspikes')
            if self.W.__class__ is SparseConnectionMatrix:
                MatrixSymbols = SparseMatrixSymbols
            elif self.W.__class__ is DenseConnectionMatrix:
                MatrixSymbols = DenseMatrixSymbols
            Value, SynapseIndex, TargetIndex = (MatrixSymbols.Value,
                                                MatrixSymbols.SynapseIndex,
                                                MatrixSymbols.TargetIndex)
            symbols['_w'] = Value(self.W, '_w', language)
            symbols['_synapse_index'] = SynapseIndex(self.W,
                                                     '_synapse_index',
                                                     '_w', language)
            symbols['_target_index'] = TargetIndex(self.W,
                                                   '_target_index',
                                                   '_w', language)
            block = Block(*statements)
            self.code = block.generate('connection', language, symbols)
            print 'CONNECTION'
            print self.code.code_str
            ns = self.code.namespace
        if self._use_codegen:
            ns = self.code.namespace
            ns['_spikes'] = spikes
            ns['_numspikes'] = len(spikes)
            ns['t'] = self.source.clock._t
            self.code()
            return
        Connection.propagate(self, spikes)

class DenseMatrixSymbols(object):
    class Value(ArraySymbol):
        def __init__(self, M, name, language, index='_synapse_index'):
            self.M = M
            ArraySymbol.__init__(self, asarray(M).reshape(-1), name,
                                 language, index=index,
                                 array_name='_flattened_'+name)
    
    class SynapseIndex(SliceIndex):
        def __init__(self, M, name, weightname, language,
                     sourceindex='_source_index', targetlen='_target_len'):
            self.M = M
            self.weightname = weightname
            self.targetlen = targetlen
            start = '({sourceindex})*({targetlen})'.format(
                    weightname=weightname, sourceindex=sourceindex,
                    targetlen=targetlen)
            end = '({sourceindex}+1)*({targetlen})'.format(
                    weightname=weightname, sourceindex=sourceindex,
                    targetlen=targetlen)
            SliceIndex.__init__(self, name, start, end, language)
        def resolve(self, read, write, vectorisable, item, namespace):
            namespace[self.targetlen] = self.M.shape[1]
            return SliceIndex.resolve(self, read, write, vectorisable, item,
                                      namespace)
    
    class TargetIndex(Symbol):
        supported_languages = ['python', 'c']
        def __init__(self, M, name, weightname, language, index='_synapse_index',
                     targetlen='_target_len'):
            self.M = M
            self.weightname = weightname
            self.index = index
            self.targetlen = targetlen
            self.sourceindex = '_source_index'
            Symbol.__init__(self, name, language)
        # Language invariant implementation
        def load(self, read, write, vectorisable):
            if self.language.name=='python' and vectorisable:
                code = '{name} = slice(None)'.format(name=self.name)
                return CodeStatement(code, set(), set())
            expr = '{index}-({sourceindex})*({targetlen})'.format(
                index=self.index, sourceindex=self.sourceindex,
                targetlen=self.targetlen)
            return MathematicalStatement(self.name, ':=', expr, dtype=int)
        def dependencies(self):
            return set([Read(self.index), Read(self.sourceindex)])

class SparseMatrixSymbols(object):
    class Value(ArraySymbol):
        def __init__(self, M, name, language, index='_synapse_index'):
            self.M = M
            ArraySymbol.__init__(self, M.alldata, name, language, index=index,
                                 array_name='_alldata_'+name)
                    
    class SynapseIndex(SliceIndex):
        def __init__(self, M, name, weightname, language, sourceindex='_source_index'):
            self.M = M
            self.weightname = weightname
            start = '_rowind_{weightname}[{sourceindex}]'.format(
                    weightname=weightname, sourceindex=sourceindex)
            end = '_rowind_{weightname}[{sourceindex}+1]'.format(
                    weightname=weightname, sourceindex=sourceindex)
            SliceIndex.__init__(self, name, start, end, language)
        def resolve(self, read, write, vectorisable, item, namespace):
            namespace['_rowind_'+self.weightname] = self.M.rowind
            return SliceIndex.resolve(self, read, write, vectorisable, item,
                                      namespace)
            
    class TargetIndex(ArraySymbol):
        def __init__(self, M, name, weightname, language, index='_synapse_index'):
            self.M = M
            self.weightname = weightname
            ArraySymbol.__init__(self, M.allj, name, language, index=index,
                                 array_name='_allj_'+weightname)

if __name__=='__main__':
    from languages import *
    from formatting import *
    tau = 10*ms
    Vt0 = 1.0
    taut = 100*ms
    eqs = Equations('''
    dV/dt = (-V+I)/tau : 1
    dI/dt = -I/tau : 1
    dVt/dt = (Vt0-Vt)/taut : 1
    ''')
    threshold = 'V>Vt'
    reset = '''
    Vt += 0.5
    V = 0
    I = 0
    '''
    G = NeuronGroup(3, eqs, threshold=threshold, reset=reset)
    G.Vt = Vt0
    
    P = PoissonGroup(1, rates=300*Hz)
    C = Connection(P, G, 'I', weight=1.0)
    C.compress()

    language = PythonLanguage()
    #language = CLanguage()

    connection_code = '''
    V += w
    '''
    
    statements = statements_from_codestring(connection_code,
                                            defined=set(['w']),
                                            eqs=eqs,
                                            infer_definitions=True)
    
    symbols = get_neuron_group_symbols(G, language, index='_target_index',
                                       subset=True)
    symbols['w'] = SparseValue(C.W, 'w', language)
    symbols['_synapse_index'] = SparseSynapseIndex(C.W, '_synapse_index', 'w',
                                                   language)
    symbols['_target_index'] = SparseTargetIndex(C.W, '_target_index', 'w',
                                                 language)
    symbols['_source_index'] = SpikeSymbol('_source_index', language)
    
    block = Block(*statements)
    code = block.generate('connection', language, symbols)
    print 'Code:\n', indent_string(code.code_str),
    print
    print 'Namespace:', code.namespace.keys()
