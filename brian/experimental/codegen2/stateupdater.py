from brian import *
from integration import *
from symbols import *
from blocks import *
from resolution import *

__all__ = ['CodeGenStateUpdater']

class CodeGenStateUpdater(StateUpdater):
    '''
    State updater using code generation, supports Python, C++, GPU.
    
    Initialised with:
    
    ``group``
        The :class:`~brian.NeuronGroup` that this will be used in.
    ``method``
        The integration method, currently one of :func:`euler`, :func:`rk2`
        or :func:`exp_euler`, but you can define your own too. See
        :func:`make_integration_step` for details.
    ``language``
        The :class:`Language` object.
        
    Creates a :class:`Block` from the equations and the ``method``, gets a set
    of :class:`Symbol` objects from :func:`get_neuron_group_symbols`, and
    defines the symbol ``_neuron_index`` as a :class:`SliceIndex`. Then calls
    :meth:`CodeItem.generate` to get the :class:`Code` object.
    
    Inserts ``t`` and ``dt`` into the namespace, and ``_num_neurons`` and
    ``_num_gpu_indices`` in case they are needed.
    '''
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
        log_info('brian.codegen2.CodeGenStateUpdater', 'STATE UPDATE CODE:\n'+self.code.code_str)
        log_info('brian.codegen2.CodeGenStateUpdater', 'STATE UPDATE NAMESPACE KEYS:\n'+str(self.code.namespace.keys()))
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
