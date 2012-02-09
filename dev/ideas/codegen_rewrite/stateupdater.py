from brian import *
from integration import *
from symbols import *
from blocks import *
from resolution import *

__all__ = ['CodeGenStateUpdater']

class CodeGenStateUpdater(StateUpdater):
    def __init__(self, group, method, language, clock=None):
        self.clock = guess_clock(clock)
        self.group = group
        eqs = group._eqs
        self.eqs = eqs
        self.method = method
        self.language = language
        block = Block(*make_integration_step(self.method, self.eqs))            
        symbols = get_neuron_group_symbols(group, self.language)
        symbols['_neuron_index'] = SliceIndex('_neuron_index',
                                              '0',
                                              '_num_neurons',
                                              self.language,
                                              all=True)
        self.code = block.generate('stateupdate', self.language, symbols)
        print 'STATE UPDATE'
        print self.code.code_str
        print 'STATE UPDATE NAMESPACE KEYS'
        print self.code.namespace.keys()
        ns = self.code.namespace
        ns['t'] = 1.0 # dummy value
        ns['dt'] = group.clock._dt
        ns['_num_neurons'] = len(group)
        ns['_num_gpu_indices'] = len(group)
    def __call__(self, G):
        code = self.code
        ns = code.namespace
        ns['t'] = G.clock._t
        code()
