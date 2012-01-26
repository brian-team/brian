from brian import *
from symbols import *
from equations import *
from statements import *
from blocks import *
from dependencies import *
from brian.inspection import namespace

__all__ = ['CodeGenThreshold']

class CodeGenThreshold(Threshold):
    def __init__(self, inputcode, language, level=0):
        inputcode = inputcode.strip()
        self.namespace = namespace(inputcode, level=level+1)
        self.inputcode = inputcode
        self.language = language
        self.prepared = False

    def __call__(self, P):
        if not self.prepared:
            ns = self.namespace
            self._inputcode = freeze_with_equations(self.inputcode, P._eqs, ns)
            block = make_threshold_block(P, self.inputcode, self.language)
            symbols = get_neuron_group_symbols(P, self.language)
            symbols['_neuron_index'] = IndexSymbol('_neuron_index',
                                                   '0',
                                                   '_num_neurons',
                                                   self.language)
            if self.language.name=='c':
                symbols['_numspikes'] = NumSpikesSymbol('_numspikes',
                                                        self.language)
            code = self.code = block.generate(self.language, symbols)
            print 'THRESHOLD'
            print self.code.code_str
            ns = self.code.namespace
            ns['dt'] = P.clock._dt
            if self.language.name=='python':
                def threshold_func(P):
                    code()
                    return ns['_spikes_bool'].nonzero()[0]
            elif self.language.name=='c':
                ns['_spikes'] = zeros(len(P), dtype=int)
                ns['_num_neurons'] = len(P)
                def threshold_func(P):
                    code()
                    return ns['_spikes'][:ns['_arr__numspikes'][0]]
            self.threshold_func = threshold_func
            self.prepared = True
        ns = self.code.namespace
        ns['t'] = P.clock._t
        return self.threshold_func(P)


def make_threshold_block(group, threshold, language):
    if language.name=='python':
        return MathematicalStatement('_spikes_bool', ':=', threshold)
    elif language.name=='c':
        return Block(
            MathematicalStatement('_spiked', ':=', threshold, boolean=True),
            CodeStatement('if(_spiked) _spikes[_numspikes++] = _neuron_index;',
                          set([Write('_spikes'), Read('_numspikes'),
                               Write('_numspikes'), Read('_neuron_index')
                               ]),
                          set())
            )

class NumSpikesSymbol(Symbol):
    def supported(self):
        return self.language.name=='c'
    def update_namespace(self, read, write, namespace):
        namespace['_arr_'+self.name] = zeros(1, dtype=int)
    def load(self, read, write):
        code = '''
            long int &{name} = _arr_{name}[0];
            {name} = 0;
            '''.format(name=self.name)
        return CodeStatement(code,
                             set([Read('_arr_'+self.name)]),
                             set())
    def dependencies(self):
        return set([Read('_arr_'+self.name)])
