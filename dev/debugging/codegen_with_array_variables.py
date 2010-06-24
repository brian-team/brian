from brian import *

#set_global_preferences(usecodegen=False)

N = 5
freq = linspace(0, 1, 5)

eqs = '''
dV/dt = cos(freq*V)/(1*second) : 1
#dV/dt = freq*(1-V)/(1*second) : 1
#freq : 1
'''

G = NeuronGroup(N, eqs)
#G.freq = freq
M = StateMonitor(G, 'V', record=True)

run(1*second)

M.plot()
show()