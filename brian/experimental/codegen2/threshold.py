from brian import *
from symbols import *
from equations import *
from statements import *
from blocks import *
from dependencies import *
from brian.inspection import namespace
from gpu import GPUCompactor

__all__ = ['CodeGenThreshold']

class CodeGenThreshold(Threshold):
    def __init__(self, group, inputcode, language, level=0):
        inputcode = inputcode.strip()
        self.namespace = namespace(inputcode, level=level+1)
        self.inputcode = inputcode
        self.language = language
        self.prepared = False
        P = group
        ns = self.namespace
        self.inputcode = freeze_with_equations(self.inputcode, P._eqs, ns)
        block = make_threshold_block(P, self.inputcode, self.language)
        symbols = get_neuron_group_symbols(P, self.language)
        symbols['_neuron_index'] = SliceIndex('_neuron_index',
                                              '0',
                                              '_num_neurons',
                                              self.language,
                                              all=True)
        if self.language.name=='c':
            symbols['_numspikes'] = NumSpikesSymbol('_numspikes',
                                                    self.language)
        if self.language.name=='gpu':
            _arr_spiked_bool = zeros(len(P)+1, dtype=int32)
            symbols['_spiked'] = ArraySymbol(_arr_spiked_bool,
                                             '_spiked',
                                             self.language,
                                             index='_neuron_index',
                                             array_name='_arr_spiked_bool',
                                             )
            self._spiked_symbol = symbols['_spiked']
            self._arr_spiked_bool = _arr_spiked_bool
        code = self.code = block.generate('threshold', self.language,
                                          symbols)
        log_info('brian.codegen2.CodeGenThreshold', 'CODE:\n'+self.code.code_str)
        log_info('brian.codegen2.CodeGenThreshold', 'KEYS:\n'+str(ns.keys()))
        ns = self.code.namespace
        ns['t'] = 1.0 # dummy value
        ns['dt'] = P.clock._dt
        ns['_num_neurons'] = len(P)
        if self.language.name=='python':
            def threshold_func(P):
                code()
                return ns['_spikes_bool'].nonzero()[0]
        elif self.language.name=='c':
            ns['_spikes'] = zeros(len(P), dtype=int)
            def threshold_func(P):
                code()
                return ns['_spikes'][:ns['_arr__numspikes'][0]]
        elif self.language.name=='gpu':
            ns['_arr_spiked_bool'] = _arr_spiked_bool
            ns['_num_gpu_indices'] = len(P)
            # TODO: this threshold func should do nothing on GPU unless
            # we want to force sync, or alternatively we can do a
            # compaction on the GPU and then return that
            compactor = GPUCompactor(len(P), index_dtype=int32)
            device_syms = code.gpu_man.mem_man.device
            def threshold_func(P):
                code()
                return compactor(device_syms['_arr_spiked_bool'])
                #if not language.force_sync:
                #    code.gpu_man.copy_to_host('_arr_spiked_bool')
                #return ns['_arr_spiked_bool'].nonzero()[0]
        self.threshold_func = threshold_func

    def __call__(self, P):
        ns = self.code.namespace
        ns['t'] = P.clock._t
        return self.threshold_func(P)


def make_threshold_block(group, threshold, language):
    if language.name=='python':
        # NOTE: this doesn't work unless the statement is vectorised, but
        # that's OK because in thresholding we're always vectorised here
        return MathematicalStatement('_spikes_bool', ':=', threshold)
    elif language.name=='gpu':
        return MathematicalStatement('_spiked', '=', threshold)
    elif language.name=='c':
        return Block(
            MathematicalStatement('_spiked', ':=', threshold, dtype=bool),
            CodeStatement('if(_spiked) _spikes[_numspikes++] = _neuron_index;',
                          set([Write('_spikes'), Read('_numspikes'),
                               Write('_numspikes'), Read('_neuron_index')
                               ]),
                          set())
            )

class NumSpikesSymbol(Symbol):
    supported_languages = ['c']
    def update_namespace(self, read, write, vectorisable, namespace):
        namespace['_arr_'+self.name] = zeros(1, dtype=int)
    def load(self, read, write, vectorisable):
        code = '''
            long int &{name} = _arr_{name}[0];
            {name} = 0;
            '''.format(name=self.name)
        return CodeStatement(code,
                             set([Read('_arr_'+self.name)]),
                             set())
    def dependencies(self):
        return set([Read('_arr_'+self.name)])
