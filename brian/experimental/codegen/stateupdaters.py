from ...stateupdater import StateUpdater
from ...globalprefs import get_global_preference
from ...clock import guess_clock
from ...log import log_debug, log_warn
from codegen_c import *
from codegen_python import *
from integration_schemes import *
import time
from scipy import weave
import numpy, scipy
import re
from c_support_code import *
try:
    import numexpr as numexpr
except ImportError:
    numexpr = None

__all__ = ['CStateUpdater', 'PythonStateUpdater']


class CStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False):
        self.clock = guess_clock(clock)
        self.eqs = eqs
        self.scheme = scheme
        self.freeze = freeze
        self.code_c = CCodeGenerator().generate(eqs, scheme)
        log_debug('brian.experimental.codegen.stateupdaters', 'C state updater code:\n' + self.code_c)
        self._weave_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._weave_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        self.namespace = {}
        code_vars = re.findall(r'\b\w+\b', self.code_c)
        self._arrays_to_check = []
        for varname in code_vars:
            if varname not in eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t', 'dt', '_S', 'num_neurons']:
                # this is kind of a hack, but since we're going to be writing a new
                # and more sensible Equations module anyway, it can stand
                for name in eqs._namespace:
                    if varname in eqs._namespace[name]:
                        varval = eqs._namespace[name][varname]
                        if isinstance(varval, numpy.ndarray):
                            self.namespace[varname] = varval
                            self._arrays_to_check.append((varname, varval))
                            self.code_c = re.sub(r'\b'+varname+r'\b', varname+'[_i]', self.code_c)
                            break

    def __call__(self, P):
        if self._arrays_to_check is not None:
            N = len(P)
            for name, X in self._arrays_to_check:
                if len(X)!=N:
                    raise ValueError('Array '+name+' has wrong size ('+str(len(X))+' instead of '+str(N)+')')
            self._arrays_to_check = None
        self.namespace['dt'] = P.clock._dt
        self.namespace['t'] = P.clock._t
        self.namespace['num_neurons'] = len(P)
        self.namespace['_S'] = P._S
        try:
            weave.inline(self.code_c, self.namespace.keys(),#['_S', 'num_neurons', 'dt', 't'],
                         local_dict = self.namespace,
                         support_code=c_support_code,
                         compiler=self._weave_compiler,
                         extra_compile_args=self._extra_compile_args)
        except:
            log_warn('brian.experimental.codegen.stateupdaters',
                     'C compilation failed, falling back on Python.')
            self.__class__ = PythonStateUpdater
            self.__init__(self.eqs, self.scheme, self.clock, self.freeze)
            self.__call__(P)


class PythonStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False):
        eqs.prepare()
        self.clock = guess_clock(clock)
        self.code_python = PythonCodeGenerator().generate(eqs, scheme)
        self.compiled_code = compile(self.code_python, 'StateUpdater code', 'exec')
        if False and numexpr is not None:
            # This only improves things for large N, in which case Python speed
            # is close to C speed anyway, so less valuable
            newcode = ''
            for line in self.code_python.split('\n'):
                m = re.search(r'(\b\w*\b\s*[^><=]?=\s*)(.*)', line) # lines of the form w = ..., w *= ..., etc.
                if m:
                    if '*' in m.group(2) or '+' in m.group(2) or '/' in m.group(2) or \
                       '-' in m.group(2) or '**' in m.group(2) or '(' in m.group(2):
                        if '[' not in m.group(2):
                            line = m.group(1) + "_numexpr.evaluate('" + m.group(2) + "')"
                newcode += line + '\n'
            self.code_python = newcode
        self.code_python = 'def _stateupdate(_S, dt, t, num_neurons):\n' + '\n'.join(['    ' + line for line in self.code_python.split('\n')]) + '\n'
        self.namespace = {'_numexpr':numexpr}
        for varname in self.compiled_code.co_names:
            if varname not in eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t', 'dt', '_S', 'num_neurons']:
                # this is kind of a hack, but since we're going to be writing a new
                # and more sensible Equations module anyway, it can stand
                for name in eqs._namespace:
                    if varname in eqs._namespace[name]:
                        self.namespace[varname] = eqs._namespace[name][varname]
                        break
                if varname not in self.namespace:
                    if hasattr(numpy, varname):
                        self.namespace[varname] = getattr(numpy, varname)
        log_debug('brian.experimental.codegen.stateupdaters', 'Python state updater code:\n' + self.code_python)
        exec self.code_python in self.namespace
        self.state_update_func = self.namespace['_stateupdate']

    def __call__(self, P):
        self.state_update_func(P._S, P.clock._dt, P.clock._t, len(P))

if __name__ == '__main__':
    from brian import *

    duration = 1 * second
    N = 10000
    domonitor = False

#    duration = 100*ms
#    N = 10
#    domonitor = True    

    eqs = Equations('''
    #dV/dt = -V*V/(10*ms) : 1
    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    dV/dt = -W*V/(100*ms) : 1
    dW/dt = -W/(100*ms) : 1
    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dW/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dW2/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dV/dt = h/(10*ms) : 1
    #h = -V*V : 1
    ''')
    print eqs

    G = NeuronGroup(N, eqs, compile=True, freeze=True, implicit=True)

    print G._state_updater.__class__

#    su = PythonStateUpdater(eqs, euler_scheme, clock=G.clock)
#    print 'Python code:'
#    print su.code_python
    su = CStateUpdater(eqs, exp_euler_scheme, clock=G.clock)
    print 'C++ loop code:'
    print su.code_c

    G.V = 1
    G.W = 0.1

    if domonitor:
        M = StateMonitor(G, 'V', record=True)

    S = copy(G._S)
    run(1 * ms)
    start = time.time()
    run(duration)
    print 'Original code:', (time.time() - start) * second
    if domonitor: M_V = M[0]
    reinit()
    G._S[:] = S
    G._state_updater = su
    run(1 * ms)
    start = time.time()
    run(duration)
    print 'New code:', (time.time() - start) * second
    if domonitor: M_V2 = M[0]

    if domonitor:
        print amax(abs(M_V - M_V2))
        plot(M.times, M_V)
        plot(M.times, M_V2)
        show()
