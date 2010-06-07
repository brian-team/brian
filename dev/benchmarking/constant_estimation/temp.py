from brian import *
import time

set_global_preferences(useweave=True)

N=4000
duration=5*second

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

Ne=int(N*0.8)
Ni=N-Ne

P=NeuronGroup(N, model=eqs,
              threshold=-48*mV)#,reset=-60*mV)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)

Ce=Connection(Pe, P, 'ge')
Ci=Connection(Pi, P, 'gi')
Ce.connect_random(Pe, P, 0.02, weight=1.62*mV)
Ci.connect_random(Pi, P, 0.02, weight=-9*mV)

M=PopulationSpikeCounter(P)

#net = MagicNetwork(verbose=False)
net=Network(P, Ce, Ci)
net.run(100*ms)

t=time.time()
net.run(duration)
print time.time()-t
print M.nspikes
raw_input('Press enter')
