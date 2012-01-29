from brian import *
from statements import *
from symbols import *
from equations import *
from blocks import *
from brian.optimiser import freeze
from brian.inspection import namespace

__all__ = ['CodeGenReset']

class CodeGenReset(Reset):
    def __init__(self, inputcode, language, level=0):
        self.namespace = namespace(inputcode, level=level+1)
        self.inputcode = inputcode
        self.language = language
        self.prepared = False

    def __call__(self, P):
        if not self.prepared:
            ns = self.namespace
            self.inputcode = freeze_with_equations(self.inputcode, P._eqs, ns)
            statements = statements_from_codestring(self.inputcode, P._eqs,
                                                    infer_definitions=True)
            block = Block(*statements)
            symbols = get_neuron_group_symbols(P, self.language,
                                               index='_neuron_index',
                                               subset=True)
            symbols['_neuron_index'] = ArrayIndex('_neuron_index',
                                                  '_spikes',
                                                   self.language,
                                                   array_len='_numspikes')
            self.code = block.generate(self.language, symbols, namespace=ns)
            print 'RESET'
            print self.code.code_str
            ns = self.code.namespace
            ns['dt'] = P.clock._dt
            self.prepared = True
        ns = self.code.namespace
        ns['_spikes'] = spikes = P.LS.lastspikes()
        ns['_numspikes'] = len(spikes)
        ns['t'] = P.clock._t
        self.code()
