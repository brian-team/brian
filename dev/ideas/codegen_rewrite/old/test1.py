from brian import *
from stateupdater import *
from threshold import *
from languages import *
from stateupdater import *
from integration import *
from reset import *

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

P = PoissonGroup(1, rates=300*Hz)
C = Connection(P, G, 'I', weight=1.0)

language = CLanguage(scalar='double')
#language = PythonLanguage()

G._state_updater = CodeGenStateUpdater(eqs, euler, language, clock=G.clock)
G._threshold = CodeGenThreshold(threshold, language)
G._resetfun = CodeGenReset(reset, language)

M = MultiStateMonitor(G, record=True)
Msp = SpikeMonitor(G)
run(100*ms)
print Msp.spikes
M.plot()
show()
