from brian import *

#set_global_preferences(usecstdp=False)

G = NeuronGroup(10, 'V:1')
C = Connection(G, G, 'V')

eqs_stdp = Equations('''
dx/dt = -x/second : 1
dy/dt = -y/second : 1
du/dt = -u/second : 1
dv/dt = -v/second : 1
''')

stdp = STDP(C, eqs=eqs_stdp, 
            pre='x+=1; y+=1; w+=u+v',
            post='u+=1; v+=1; w+=x+y', 
            wmax=10)
