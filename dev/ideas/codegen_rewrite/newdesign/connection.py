from brian import *
from symbols import *
from blocks import *
from statements import *
from dependencies import *

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
                                               index='_target_index',
                                               subset=True)
            if modulation:
                src_symbols = get_neuron_group_symbols(self.source, language,
                                               index='_source_index',
                                               subset=True,
                                               prefix='_sourcevar_')
                symbols.update(src_symbols)
            symbols['_w'] = SparseValue(self.W, '_w', language)
            symbols['_synapse_index'] = SparseSynapseIndex(self.W,
                                                           '_synapse_index',
                                                           '_w', language)
            symbols['_target_index'] = SparseTargetIndex(self.W,
                                                         '_target_index',
                                                         '_w', language)
            symbols['_source_index'] = SpikeSymbol('_source_index', language)
            block = Block(*statements)
            self.code = block.generate(language, symbols)
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


class SparseValue(ArraySymbol):
    def __init__(self, M, name, language, index='_synapse_index'):
        self.M = M
        ArraySymbol.__init__(self, M.alldata, name, language, index=index,
                             subset=True, array_name='_alldata_'+name)
                
class SparseSynapseIndex(IndexSymbol):
    def __init__(self, M, name, weightname, language, sourceindex='_source_index'):
        self.M = M
        self.weightname = weightname
        start = '_rowind_{weightname}[{sourceindex}]'.format(
                weightname=weightname, sourceindex=sourceindex)
        end = '_rowind_{weightname}[{sourceindex}+1]'.format(
                weightname=weightname, sourceindex=sourceindex)
        IndexSymbol.__init__(self, name, start, end, language)
    def resolve(self, read, write, item, namespace):
        namespace['_rowind_'+self.weightname] = self.M.rowind
        return IndexSymbol.resolve(self, read, write, item, namespace)
        
class SparseTargetIndex(ArraySymbol):
    def __init__(self, M, name, weightname, language, index='_synapse_index'):
        self.M = M
        self.weightname = weightname
        ArraySymbol.__init__(self, M.allj, name, language, index=index,
                             subset=True, array_name='_allj_'+weightname,
                             readname=name)

class SpikeSymbol(IndexSymbol):
    def __init__(self, name, language, index_array='_spikes',
                 start='0', end='_numspikes'):
        IndexSymbol.__init__(self, name, start, end, language,
                             index_array=index_array)
    def resolve(self, read, write, item, namespace):
        if self.language.name=='python':
            return PythonForBlock(self.name, self.index_array, item)
        else:
            return IndexSymbol.resolve(self, read, write, item, namespace)

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
    code = block.generate(language, symbols)
    print 'Code:\n', indent_string(code.code_str),
    print
    print 'Namespace:', code.namespace.keys()
