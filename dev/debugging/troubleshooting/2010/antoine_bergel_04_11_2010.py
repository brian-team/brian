'''
Bug with insert_spikes, filed by Antoine Bergel on 4 Nov 2010
'''
from brian import *

G=NeuronGroup(1,model='dv/dt=100*Hz:1',threshold=1,reset=0)
M = SpikeMonitor(G)
Mv = StateMonitor(G, 'v', record=True)

run(100*ms)
print Mv.values.shape
Mv.insert_spikes(M,5)

plot(Mv.times/ms, Mv[0])
show()
