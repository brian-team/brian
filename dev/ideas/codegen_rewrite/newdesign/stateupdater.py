from brian import *
from integration import *
from symbols import *
from blocks import *
from resolution import *

__all__ = ['CodeGenStateUpdater']

class CodeGenStateUpdater(StateUpdater):
    def __init__(self, eqs, method, language, clock=None):
        self.clock = guess_clock(clock)
        self.eqs = eqs
        self.method = method
        self.language = language
        self.prepared = False
    def __call__(self, G):
        if not self.prepared:
            block = Block(*make_integration_step(self.method, self.eqs))            
            symbols = get_neuron_group_symbols(G, self.language)
            symbols['_neuron_index'] = IndexSymbol('_neuron_index',
                                                   '0',
                                                   '_num_neurons',
                                                   self.language)
            self.code = block.generate(self.language, symbols)
            print 'STATE UPDATE'
            print self.code.code_str
            ns = self.code.namespace
            ns['dt'] = G.clock._dt
            ns['_num_neurons'] = len(G)
            self.prepared = True
        code = self.code
        ns = code.namespace
        ns['t'] = G.clock._t
        code()
