from brian import *

tau=10*ms
bof=3.0

eqs=Equations('''
dx/dt = -x/tau : 1
V = x*bof : 1
''')

G=NeuronGroup(1, eqs)
G2=NeuronGroup(1, eqs)

G.x=1
G2.x=2

print G.V
print G2.V
