from brian import *
from brian.experimental.codegen.reset import *

eqs = Equations('''
#dV/dt = -V/(100*second) : volt
V : volt
Vt : volt
''')

dVt = 2 * mV

reset = '''
V = 0*volt
Vt += dVt
'''

print '**** C CODE *****'
print generate_c_reset(eqs, reset)

print '**** PYTHON CODE ****'
print generate_python_reset(eqs, reset)

GP = NeuronGroup(10, eqs, threshold=1 * volt,
                reset=PythonReset(reset))
GC = NeuronGroup(10, eqs, threshold=1 * volt,
                reset=CReset(reset))

GP.V = 0.5
GP.Vt = 0.1
GP.V[[2, 4, 8]] = 2
GC.V = 0.5
GC.Vt = 0.1
GC.V[[2, 4, 8]] = 2
run(1 * ms)
print 'GP.V  =', GP.V
print 'GP.Vt =', GP.Vt
print 'GC.V  =', GC.V
print 'GC.Vt =', GC.Vt
