import re
from rewriting import *
from ...inspection import namespace
from ...optimiser import freeze
from ...reset import Reset
from ...equations import Equations
from ...globalprefs import get_global_preference
from ...log import log_warn
from expressions import *
from scipy import weave
from c_support_code import *

__all__ = ['generate_c_reset', 'generate_python_reset',
           'CReset', 'PythonReset']

def generate_c_reset(eqs, inputcode, vartype='double', level=0, ns=None):
    if ns is None:
        ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
    all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    code = ''
    for j, name in enumerate(eqs._diffeq_names):
        code += vartype + ' *' + name + '__Sbase = _S+' + str(j) + '*_num_neurons;\n'
    code += 'for(int _i=0;_i<_nspikes;_i++){\n'
    code += '    long _j = _spikes[_i];\n'
    for j, name in enumerate(eqs._diffeq_names):
        code += '    ' + vartype + ' &' + name + ' = ' + name + '__Sbase[_j];\n'
    for line in inputcode.split('\n'):
        line = line.strip()
        if line:
            line = freeze(line, all_variables, ns)
            line = c_single_statement(line)
            code += '    ' + line + '\n'
    code += '}\n'
    return code

def generate_python_reset(eqs, inputcode, level=0, ns=None):
    if ns is None:
        ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
    all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t', '_spikes']
    code = ''
    for var in eqs._diffeq_names:
        inputcode = re.sub("\\b" + var + "\\b", var + '[_spikes]', inputcode)
    for line in inputcode.split('\n'):
        line = line.strip()
        if line:
            line = freeze(line, all_variables, ns)
            line = python_single_statement(line)
            code += line + '\n'
    return code


class PythonReset(Reset):
    def __init__(self, inputcode, level=0):
        self._ns, unknowns = namespace(inputcode, level=level + 1, return_unknowns=True)
        self._inputcode = inputcode
        self._prepared = False

    def __call__(self, P):
        ns = self._ns
        if not self._prepared:
            vars = [var for var in P.var_index if isinstance(var, str)]
            eqs = P._eqs
            outputcode = generate_python_reset(eqs, self._inputcode, ns=ns)
            self._compiled_code = compile(outputcode, "PythonReset", "exec")
            for var in vars:
                ns[var] = P.state(var)
            self._prepared = True
        cc = self._compiled_code
        spikes = P.LS.lastspikes()
        ns['_spikes'] = spikes
        exec cc in ns


class CReset(Reset):
    def __init__(self, inputcode, level=0):
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
            self._outputcode = generate_c_reset(eqs, self._inputcode, ns=self._ns)
        _spikes = P.LS.lastspikes()
        dt = P.clock._dt
        t = P.clock._t
        _nspikes = len(_spikes)
        _S = P._S
        _num_neurons = len(P)
        try:
            weave.inline(self._outputcode, ['_S', '_nspikes', 'dt', 't', '_spikes', '_num_neurons'],
                         c_support_code=c_support_code,
                         compiler=self._weave_compiler,
                         extra_compile_args=self._extra_compile_args)
        except:
            log_warn('brian.experimental.codegen.reset',
                     'C compilation failed, falling back on Python.')
            self.__class__ = PythonReset
            self._prepared = False
            return self.__call__(P)
