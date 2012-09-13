#!/usr/bin/env python
'''
Multiple delays
'''
from brian import *

P=NeuronGroup(1,model='dv/dt=1/(20*ms):1',threshold=1,reset=0)
Q=NeuronGroup(1,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w')
M=StateMonitor(Q,'v',record=True)

S[0,0]=2
S.w[0,0,0]=1.
S.w[0,0,1]=.5
S.delay[0,0,0]=.5*ms
S.delay[0,0,1]=1*ms

run(60*ms)

plot(M.times/ms,M[0])
show()
