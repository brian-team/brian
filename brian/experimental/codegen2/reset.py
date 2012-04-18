from brian import *
from statements import *
from symbols import *
from equations import *
from blocks import *
from brian.optimiser import freeze
from brian.inspection import namespace

__all__ = ['CodeGenReset']

class CodeGenReset(Reset):
    def __init__(self, group, inputcode, language, level=0):
        self.namespace = namespace(inputcode, level=level+1)
        self.inputcode = inputcode
        self.language = language
        P = group
        ns = self.namespace
        self.inputcode = freeze_with_equations(self.inputcode, P._eqs, ns)
        statements = statements_from_codestring(self.inputcode, P._eqs,
                                                infer_definitions=True)
        block = Block(*statements)
        symbols = get_neuron_group_symbols(P, self.language,
                                           index='_neuron_index')
        if language.name=='python' or language.name=='c':
            symbols['_neuron_index'] = ArrayIndex('_neuron_index',
                                                  '_spikes',
                                                   self.language,
                                                   array_len='_numspikes')
        if language.name=='gpu':
            symbols['_neuron_index'] = SliceIndex('_neuron_index',
                                                  '0',
                                                  '_num_neurons',
                                                  self.language,
                                                  all=True)
            _spiked_symbol = group._threshold._spiked_symbol
            symbols['_spiked'] = _spiked_symbol
            block = CIfBlock('_spiked', [block])
            self.namespace['_num_gpu_indices'] = len(P)
            self.namespace['t'] = 1.0 # dummy value
            self.namespace['_num_neurons'] = len(P)
        self.code = block.generate('reset', self.language, symbols,
                                   namespace=ns)
        ns = self.code.namespace
        ns['dt'] = P.clock._dt
        log_info('brian.codegen2.CodeGenReset', 'CODE:\n'+self.code.code_str)
        log_info('brian.codegen2.CodeGenReset', 'KEYS:\n'+str(ns.keys()))

    def __call__(self, P):
        ns = self.code.namespace
        ns['_spikes'] = spikes = P.LS.lastspikes()
        ns['_numspikes'] = len(spikes)
        ns['t'] = P.clock._t
        self.code()
