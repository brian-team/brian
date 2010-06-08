from brian import *
from brian.experimental.codegen.threshold import *

eqs = Equations('''
V : volt
Vt : volt
''')

threshold = '''
    (V>Vt)&(V>Vt)
'''

print '**** C CODE *****'
print generate_c_threshold(eqs, threshold)
#
print '**** PYTHON CODE ****'
print generate_python_threshold(eqs, threshold)

GP = NeuronGroup(10, eqs, threshold=PythonThreshold(threshold), reset=0 * volt)
GC = NeuronGroup(10, eqs, threshold=CThreshold(threshold), reset=0 * volt)
MP = SpikeMonitor(GP)
MC = SpikeMonitor(GC)

GP.V[[1, 2, 4, 8]] = 2
GP.Vt = 1 * volt
GP.Vt[[2, 8]] = 3
GC.V[[1, 2, 4, 8]] = 2
GC.Vt = 1 * volt
GC.Vt[[2, 8]] = 3

run(1 * ms)

print GP.V
print GP.Vt
print MP.spikes
print
print GC.V
print GC.Vt
print MC.spikes
