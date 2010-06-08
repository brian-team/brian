from brian import *

eqs = '''
dV/dt = (0.5-V)/(10*ms) : 1
refrac : second
'''

#G = NeuronGroup(10, eqs, reset=0, threshold=1, refractory=0.5*ms)
#G = NeuronGroup(10, eqs, reset=0, threshold=1, refractory=linspace(0*ms, 5*ms, 10, endpoint=False))
G = NeuronGroup(10, eqs, reset=0, threshold=1, refractory='refrac', max_refractory=10 * ms)
G.refrac = linspace(0 * ms, 5 * ms, 10, endpoint=False)
G.V = 2

M = StateMonitor(G, 'V', record=True)

run(1 * ms)

#print G._next_allowed_spiketime 

run(9 * ms)

G.V = 2
#G.refrac = linspace(2*ms, 0*ms, 10, endpoint=False)

run(5 * ms)

M.plot()
show()
