from ...stateupdater import StateUpdater
from ...globalprefs import get_global_preference
from ...clock import guess_clock
from codegen_c import *
from codegen_python import *
from integration_schemes import *
import time
from scipy import weave
import numpy, scipy

__all__ = ['CStateUpdater', 'PythonStateUpdater']

class CStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False):
        self.clock = guess_clock(clock)
        self.code_c = CCodeGenerator().generate(eqs, scheme)
        self._weave_compiler = get_global_preference('weavecompiler')
    def __call__(self, P):
        dt = P.clock._dt
        t = P.clock._t
        num_neurons = len(P)
        _S = P._S
        weave.inline(self.code_c, ['_S', 'num_neurons', 'dt', 't'],
                     compiler=self._weave_compiler,
                     extra_compile_args=['-O3'])

class PythonStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False):
        eqs.prepare()
        self.clock = guess_clock(clock)
        self.code_python = PythonCodeGenerator().generate(eqs, scheme)
        self.compiled_code = compile(self.code_python, 'StateUpdater code', 'exec')
        self.code_python = 'def _stateupdate(_S, dt, t, num_neurons):\n' + '\n'.join(['    '+line for line in self.code_python.split('\n')]) + '\n'
        self.namespace = {}
        for varname in self.compiled_code.co_names:
            if varname not in eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t', 'dt', '_S', 'num_neurons']:
                if hasattr(numpy, varname):
                    self.namespace[varname] = getattr(numpy, varname)
        exec self.code_python in self.namespace
        self.state_update_func = self.namespace['_stateupdate']
    def __call__(self, P):
        self.state_update_func(P._S, P.clock._dt, P.clock._t, len(P))
       
if __name__=='__main__':
    from brian import *
    
    duration = 1*second
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
    run(1*ms)
    start = time.time()
    run(duration)
    print 'Original code:', (time.time()-start)*second
    if domonitor: M_V = M[0]
    reinit()
    G._S[:] = S
    G._state_updater = su
    run(1*ms)
    start = time.time()
    run(duration)
    print 'New code:', (time.time()-start)*second
    if domonitor: M_V2 = M[0]
    
    if domonitor:
        print amax(abs(M_V-M_V2))
        plot(M.times, M_V)
        plot(M.times, M_V2)
        show()