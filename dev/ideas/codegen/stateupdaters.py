from brian import *
from codegen_c import *
from codegen_python import *
from integration_schemes import *
import time
from scipy import weave

class CStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False):
        self.clock = guess_clock(clock)
        self.code_c = CCodeGenerator().generate(eqs, scheme)
    def __call__(self, P):
        dt = float(P.clock._dt)
        t = float(P.clock.t)
        num_neurons = len(P)
        _S = P._S
        weave.inline(self.code_c, ['_S', 'num_neurons', 'dt', 't'],
                     compiler='gcc',
                     #type_converters=weave.converters.blitz,
                     extra_compile_args=['-O3'])#O2 seems to be faster than O3 here

class PythonStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False):
        self.clock = guess_clock(clock)
        self.code_python = PythonCodeGenerator().generate(eqs, scheme)
        self.compiled_code = compile(self.code_python, 'StateUpdater code', 'exec')
        self.namespace = {}
    def __call__(self, P):
        self.namespace['_S'] = P._S
        self.namespace['dt'] = float(P.clock._dt)
        self.namespace['t'] = float(P.clock.t)
        self.namespace['num_neurons'] = len(P)
        exec self.compiled_code in self.namespace
        
if __name__=='__main__':
    
#    duration = 1*second
#    N = 1000
#    domonitor = False
    
    duration = 100*ms
    N = 10
    domonitor = True
    
    eqs = Equations('''
    #dV/dt = -V*V/(10*ms) : 1
    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    dV/dt = W*W/(100*ms) : 1
    dW/dt = -V/(100*ms) : 1
    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dW/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dW2/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dV/dt = h/(10*ms) : 1
    #h = -V*V : 1
    ''')
    print eqs
    
    G = NeuronGroup(N, eqs, compile=True, freeze=True)
    

#    su = PythonStateUpdater(eqs, euler_scheme, clock=G.clock)
#    print 'Python code:'
#    print su.code_python
    su = CStateUpdater(eqs, euler_scheme, clock=G.clock)
    print 'C++ loop code:'
    print su.code_c
    
    G.V = 1
    
    if domonitor:
        M = StateMonitor(G, 'V', record=True)
    
    S = copy(G._S)
    start = time.time()
    run(duration)
    print 'Original code:', (time.time()-start)*second
    if domonitor: M_V = M[0]
    reinit()
    G._S[:] = S
    G._state_updater = su
    start = time.time()
    run(duration)
    print 'New code:', (time.time()-start)*second
    if domonitor: M_V2 = M[0]
    
    if domonitor:
        print amax(abs(M_V-M_V2))
        plot(M.times, M_V)
        plot(M.times, M_V2)
        show()