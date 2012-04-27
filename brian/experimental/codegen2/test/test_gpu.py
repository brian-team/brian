from brian import *
from brian.experimental.codegen2 import *
from brian.experimental.codegen2.gpu import *

log_level_info()

#language = GPULanguage(scalar='float')
language = GPULanguage()

structure = 'sparse'
#structure = 'dense'

tau = 10*ms
Vt0 = 1.0
taut = 100*ms
eqs = Equations('''
dV/dt = (-V+I)/tau : 1
dI/dt = -I/tau : 1
dVt/dt = (Vt0-Vt)/taut : 1
''')
threshold = 'V>Vt'
reset = '''
Vt += 0.5
V = 0
I = 0
'''
G = NeuronGroup(3, eqs, threshold=threshold, reset=reset)

G.Vt = Vt0

H = NeuronGroup(len(G), 'V:1\nmod:1', reset=0, threshold=1)
P = PoissonGroup(1, rates=300*Hz)
Ci = Connection(P, H, 'V', weight=2)
H.mod = [1.0, 0.9, 0.1]

C = Connection(H, G, 'I', modulation='mod', structure=structure)
for i in xrange(len(G)):
    C[i, i] = 1

G._state_updater = CodeGenStateUpdater(G, euler, language, clock=G.clock)
G._threshold = CodeGenThreshold(G, threshold, language)
G._resetfun = CodeGenReset(G, reset, language)

M = MultiStateMonitor(G, record=True)
Msp = SpikeMonitor(G)

#language.gpu_man.prepare()
#exit()

run(100*ms)
print Msp.spikes
M.plot()
show()

