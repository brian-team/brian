'''
One synapse within several possibilities.
Synapse from 0->2,3.

Works in non simultaneous and identical delays.
Also works with 0->2 and 1->3
'''
from brian import *
from dev.ideas.synapses.synapses import *

log_level_debug()

P=NeuronGroup(2,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(4,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w',max_delay=1*ms)
M=StateMonitor(Q,'v',record=True)

S.synapses_pre[0]=array([0,1],dtype=S.synapses_pre[0].dtype)
S.w[0:2]=[1.,.7]
S.delay_pre[0:2]=[5,7] # in timebins
S.presynaptic[0:2]=[0,0]
S.postsynaptic[0:2]=[2,3]

run(40*ms)

for i in range(4):
    plot(M.times/ms,M[i]+i*2,'k')
show()
