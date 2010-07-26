from brian import *

set_global_preferences(usecstdp=False)

G = NeuronGroup(10, 'V:1')
C = Connection(G, G, 'V')

eqs_stdp = Equations('''
dx/dt = -x/second : 1
y = x*x : 1
du/dt = -u/second : 1
v = u*u : 1
''')

stdp = STDP(C, eqs=eqs_stdp, 
            pre='x+=1; w+=v',
            post='u+=1; w+=y', 
            wmax=10)
