from brian import *

eqs = '''
dV/dt = -V/(10*ms)+xi/(10*ms)**.5 : 1
W : 1
'''

G = NeuronGroup(1, eqs)
#G.link_var('W', G, 'V')
G.W = linked_var(G, 'V', func=lambda x:x+.1)
M = StateMonitor(G, 'V', record=True, when='before_groups')
MW = StateMonitor(G, 'W', record=True, when='before_groups')

run(100*ms)

M.plot()
MW.plot()
show()