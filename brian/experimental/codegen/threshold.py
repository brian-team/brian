import re
from rewriting import *
from ...inspection import namespace
from ...optimiser import freeze
from ...threshold import Threshold
from ...equations import Equations
from ...globalprefs import get_global_preference
from ...log import log_warn
from expressions import *
from scipy import weave
from c_support_code import *

__all__ = ['generate_c_threshold', 'generate_python_threshold',
           'CThreshold', 'PythonThreshold']

def generate_c_threshold(eqs, inputcode, vartype='double', level=0, ns=None):
    inputcode = inputcode.strip()
    if ns is None:
        ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
    all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    code = 'int _numspikes = 0;\n'
    for j, name in enumerate(eqs._diffeq_names):
        code += vartype + ' *' + name + '__Sbase = _S+' + str(j) + '*_num_neurons;\n'
    code += 'for(int _i=0;_i<_num_neurons;_i++){\n'
    for j, name in enumerate(eqs._diffeq_names):
        code += '    ' + vartype + ' &' + name + ' = ' + name + '__Sbase[_i];\n'
    inputcode = freeze(inputcode.strip(), all_variables, ns)
    inputcode = c_single_expr(inputcode)
    code += '    if(' + inputcode + ')\n'
    code += '        _spikes[_numspikes++] = _i;\n'
    code += '}\n'
    code += 'return_val = _numspikes;\n'
    return code

def generate_python_threshold(eqs, inputcode, level=0, ns=None):
    inputcode = inputcode.strip()
    if ns is None:
        ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
    all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    inputcode = inputcode.strip()
    inputcode = freeze(inputcode, all_variables, ns)
    inputcode = python_single_expr(inputcode)
    return inputcode


class PythonThreshold(Threshold):
    def __init__(self, inputcode, level=0):
        inputcode = inputcode.strip()
        self._ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
        self._inputcode = inputcode
        self._prepared = False

    def __call__(self, P):
        if not self._prepared:
            ns = self._ns
            vars = [var for var in P.var_index if isinstance(var, str)]
            eqs = P._eqs
            outputcode = generate_python_threshold(eqs, self._inputcode, ns=ns)
            for var in vars:
                ns[var] = P.state(var)
            self._compiled_code = compile(outputcode, "PythonThreshold", "eval")
            self._prepared = True
        return eval(self._compiled_code, self._ns).nonzero()[0]


class CThreshold(Threshold):
    def __init__(self, inputcode, level=0):
        inputcode = inputcode.strip()
        self._ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
        self._inputcode = inputcode
        self._prepared = False
        self._weave_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._weave_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']

    def __call__(self, P):
        if not self._prepared:
            vars = [var for var in P.var_index if isinstance(var, str)]
            eqs = P._eqs
            self._outputcode = generate_c_threshold(eqs, self._inputcode, ns=self._ns)
        _spikes = P._spikesarray
        t = P.clock._t
        _S = P._S
        _num_neurons = len(P)
        try:
            _numspikes = weave.inline(self._outputcode,
                                      ['_S', 't', '_spikes', '_num_neurons'],
                                      support_code=c_support_code,
                                      compiler=self._weave_compiler,
                                      extra_compile_args=self._extra_compile_args)
        except:
            log_warn('brian.experimental.codegen.threshold',
                     'C compilation failed, falling back on Python.')
            self.__class__ = PythonThreshold
            self._prepared = False
            return self.__call__(P)
        return _spikes[0:_numspikes]

