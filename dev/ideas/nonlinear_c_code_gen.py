from brian import *
import brian.optimiser as optimiser
from scipy import weave

eqs = Equations('''
#dV/dt = -V*V/(10*ms) : 1
#dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
dV/dt = W*W/(100*ms) : 1
dW/dt = -V/(100*ms) : 1
''')

G = NeuronGroup(1, eqs, compile=True, freeze=True)

class AutoCompiledNonlinearStateUpdater(NonlinearStateUpdater):
    def __init__(self,eqs,clock=None,freeze=False):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.code_python, self.code_c, self.code_c_vars = self.generate_forward_euler_code()
        self.code_python_compiled = compile(self.code_python,'Euler update code','exec')
    def generate_forward_euler_code(self):
        eqs = self.eqs
        all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
        vars_tmp = [name+'__tmp' for name in eqs._diffeq_names]
        pythonlines = ', '.join([name+'__arr' for name in eqs._diffeq_names])+'=P._S\n'
        pythonlines += ', '.join([name+'__arr' for name in vars_tmp])+'=P._dS\n'
        clines = 'for(int i=0;i<n;i++){\n'
        cvars = [name+'__arr' for name in eqs._diffeq_names+vars_tmp] + ['n','dt','t']
        for name in eqs._diffeq_names+vars_tmp:
            clines += '    double &'+name+' = '+name+'__arr(i);\n'
        for name in eqs._diffeq_names_nonzero:
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            clines += '    '+name+'__tmp='+expr+';\n'
        for name in eqs._diffeq_names:
            clines += '    '+name+' += dt*'+name+'__tmp;\n'
        clines += '}\n'
        return (pythonlines, clines, cvars)
    def __call__(self, P):
        if self._first_time:
            self._first_time = False
            P._dS = 0*P._S
        dt = float(P.clock._dt)
        t = float(P.clock.t)
        n = len(P)
        exec(self.code_python_compiled)
        weave.inline(self.code_c, self.code_c_vars,
                     compiler='gcc',
                     type_converters=weave.converters.blitz,
                     extra_compile_args=['-O2'])#O2 seems to be faster than O3 here

su = AutoCompiledNonlinearStateUpdater(eqs, G.clock, freeze=True)
print 'Python initialisation code:'
print su.code_python
print 'C++ loop code:'
print su.code_c
print 'C++ variables:'
print su.code_c_vars

G.V = 1

M = StateMonitor(G, 'V', record=True)

S = copy(G._S)
run(100*ms)
M_V = M[0]
reinit()
G._S[:] = S
G._state_updater = su
run(100*ms)
M_V2 = M[0]

plot(M.times, M_V)
plot(M.times, M_V2)
show()