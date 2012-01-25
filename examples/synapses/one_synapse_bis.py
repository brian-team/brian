'''
One synapse within several possibilities.
Synapse from 2->3.
'''
from brian import *
from brian.experimental.synapses import *

#log_level_debug()

P=NeuronGroup(5,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(4,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w')
M=StateMonitor(Q,'v',record=True)
#P.v[2]=.5

S[2,3]=True
S.w[2,3]=1.
S.delay[2,3]=.5*ms

run(40*ms)

for i in range(4):
    plot(M.times/ms,M[i]+i*2,'k')
show()
