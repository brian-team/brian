from brian import *

eqs1 = '''
dV/dt = -V/(10*ms) : 1
w = 10*V : 1
'''

eqs2 = '''
V : 1
w = 10*V : 1
'''

G1 = NeuronGroup(1, eqs1)
G2 = NeuronGroup(1, eqs2)

G1.V = 1
G2.V = linked_var(G1, 'w')

print G1.staticvars
print G2.staticvars

M = StateMonitor(G2, 'V', record=True)

run(30*ms)

M.plot()
show()
