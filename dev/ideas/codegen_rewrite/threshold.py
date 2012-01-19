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

__all__ = ['CodeGenThreshold']


class CodeGenThreshold(Threshold):
    def __init__(self, inputcode, language, level=0):
        inputcode = inputcode.strip()
        self._ns, unknowns = namespace(inputcode, level=level+1, return_unknowns=True)
        self._inputcode = inputcode
        self._language = language
        self._prepared = False

    def __call__(self, P):
        if not self._prepared:
            ns = self._ns
            self._inputcode = freeze_with_equations(self._inputcode, P._eqs, ns)
            self._block = make_threshold_code_block(P, self._inputcode, self._language)
            self._symbols = get_neuron_group_symbols(P, self._language)
            code = self._code = self._block.generate(self._language, self._symbols)
            ns = self._code.namespace
            if self._language.name=='python':
                def threshold_func(P):
                    code()
                    return ns['_spikes_bool'].nonzero()[0]
            elif self._language.name=='c':
                self._code.namespace['_spikes'] = zeros(len(P), dtype=int)
                self._code.namespace['_numspikes_arr'] = zeros(1, dtype=int)
                self._code.namespace['_num_neurons'] = len(P)
                def threshold_func(P):
                    code()
                    return ns['_spikes'][:ns['_numspikes_arr'][0]]
            self._threshold_func = threshold_func
            self._prepared = True
        return self._threshold_func(P)
    

def make_threshold_code_block(group, threshold, language):
    if language.name=='python':
        expr = Expression(threshold)
        stmt = Statement('_spikes_bool', ':=', expr)
        return CodeBlock([], [stmt])
    elif language.name=='c':
        expr = Expression(threshold)
        stmt = Statement('_spiked', ':=', expr, boolean=True)
        threshold_step = CodeBlock(['_neuron_index'], [stmt])
        loop_block = CodeBlock([], [
            CodeBlock([],
            '''
            long int &_numspikes = _numspikes_arr[0];
            _numspikes = 0;
            for(int _neuron_index=0; _neuron_index<_num_neurons; _neuron_index++)
            {
            '''),
                threshold_step.indented(),
            CodeBlock([], '''
                if(_spiked)
                {
                    _spikes[_numspikes++] = _neuron_index;
                }
            }
            '''),
            ])
        return loop_block
