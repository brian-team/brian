from brian import *

eqs = '''
dV/dt = -V/(100*ms) : 1
Vt : 1
Vr : 1
'''

const = -1.1

G = NeuronGroup(2, eqs, threshold='V>Vt', reset='V=-Vt*Vr*1.1')

print G._threshold
print G._resetfun

G.V = -1
G.Vt = [-0.5, -0.75]
G.Vr = [-1, -2]

M = MultiStateMonitor(G, record=True)
run(400*ms)
M.plot()
legend()
show()