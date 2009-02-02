from brian import *
import brian.optimiser as optimiser
from scipy import weave
import time

class AutoCompiledNonlinearStateUpdater(NonlinearStateUpdater):
    def __init__(self,eqs,clock=None,freeze=False):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.code_c = self.generate_forward_euler_code()
    def generate_forward_euler_code(self):
        eqs = self.eqs
        all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
        clines = ''
        for j, name in enumerate(eqs._diffeq_names):
            clines += 'double *' + name + '__Sbase = S+'+str(j)+'*n;\n'
        clines += 'for(int i=0;i<n;i++){\n'
        for j, name in enumerate(eqs._diffeq_names):
            clines += '    double &'+name+' = *'+name+'__Sbase++;\n'
        for name in eqs._eq_names:
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            clines += '    double '+name+'__tmp = '+expr+';\n'
        for j, name in enumerate(eqs._diffeq_names):
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            if name in eqs._diffeq_names_nonzero:
                clines += '    double '+name+'__tmp = '+expr+';\n'
        for name in eqs._diffeq_names_nonzero:
            clines += '    '+name+' += dt*'+name+'__tmp;\n'
        clines += '}\n'
        #print clines
        return clines
    def __call__(self, P):
        dt = float(P.clock._dt)
        t = float(P.clock.t)
        n = len(P)
        S = P._S
        weave.inline(self.code_c, ['S', 'n', 'dt', 't'],
                     compiler='gcc',
                     #type_converters=weave.converters.blitz,
                     extra_compile_args=['-O3'])#O2 seems to be faster than O3 here

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
    
    su = AutoCompiledNonlinearStateUpdater(eqs, G.clock, freeze=True)
    print 'Python code generated for by Python compilation mechanism:'
    print eqs.forward_euler_code_string()
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
        plot(M.times, M_V)
        plot(M.times, M_V2)
        show()