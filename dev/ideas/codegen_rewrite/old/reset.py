from brian import *
from languages import *
from expressions import *
from formatter import *
from symbols import *
from codeobject import *
from generation import *
from equations import *
from brian.optimiser import freeze
from brian.inspection import namespace

__all__ = ['CodeGenReset']

class ResetSymbol(NeuronGroupStateVariableSymbol):
    @property
    def read(self):
        if self.language.name=='python':
            return self.name+'['+self.index_name+']'
        elif self.language.name=='c':
            return self.name
    write = read

def get_reset_symbols(group, language, index_name='_neuron_index'):
    eqs = group._eqs
    symbols = dict(
       (name,
        ResetSymbol(group, name, name, language,
                    index_name=index_name)) for name in eqs._diffeq_names)
    return symbols

class CodeGenReset(Reset):
    def __init__(self, inputcode, language, level=0):
        self._ns, unknowns = namespace(inputcode, level=level+1, return_unknowns=True)
        self._inputcode = inputcode
        self._language = language
        self._prepared = False

    def __call__(self, P):
        if not self._prepared:
            ns = self._ns
            self._inputcode = freeze_with_equations(self._inputcode, P._eqs, ns)
            self._block = make_reset_code_block(P, P._eqs, self._inputcode,
                                                self._language)
            if self._language.name=='python':
                self._symbols = get_reset_symbols(P, self._language,
                                                  index_name='_spikes')
            elif self._language.name=='c':
                self._symbols = get_reset_symbols(P, self._language)
            code = self._code = self._block.generate(self._language, self._symbols)
            ns = self._code.namespace
            ns['dt'] = P.clock._dt
            self._prepared = True
        ns = self._code.namespace
        ns['_spikes'] = spikes = P.LS.lastspikes()
        ns['_num_spikes'] = len(spikes)
        ns['t'] = P.clock._t
        return self._code()
    
def make_reset_code_block(group, eqs, reset, language):
    reset_step = CodeBlock(['_neuron_index', '_spikes'],
                           statements_from_codestring(reset, eqs,
                                                      infer_definitions=True))
    if language.name=='python':
        return reset_step
    elif language.name=='c':
        loop_block = CodeBlock([], [
            CodeBlock([],
            '''
            for(int _spike_index=0; _spike_index<_num_spikes; _spike_index++)
            {
                int _neuron_index = _spikes[_spike_index];
            '''),
                reset_step.indented(),
            CodeBlock([], '''
            }
            '''),
            ])
        return loop_block
