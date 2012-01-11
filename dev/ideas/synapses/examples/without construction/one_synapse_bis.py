'''
One synapse within several possibilities.
Synapse from 2->3.
'''
from brian import *
from dev.ideas.synapses.synapses import *

log_level_debug()

P=NeuronGroup(5,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(4,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w',max_delay=1*ms)
M=StateMonitor(Q,'v',record=True)
#P.v[2]=.5

S.synapses_pre[2]=array([0],dtype=S.synapses_pre[0].dtype)
S.w[0]=1.
S.delay_pre[0]=5 # in timebins
S.presynaptic[0]=2
S.postsynaptic[0]=3

run(40*ms)

for i in range(4):
    plot(M.times/ms,M[i]+i*2,'k')
show()
