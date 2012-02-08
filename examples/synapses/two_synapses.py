'''
One synapse within several possibilities.
Synapse from 0->2,3.
'''
from brian import *
from brian.experimental.synapses import *

#log_level_debug()

P=NeuronGroup(2,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(4,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w')
M=StateMonitor(Q,'v',record=True)

S[0,2]=True
S[0,3]=True
S.w[0,:]=[1.,.7]
S.delay[0,:]=[.5*ms,.7*ms]

run(40*ms)

for i in range(4):
    plot(M.times/ms,M[i]+i*2,'k')
show()
