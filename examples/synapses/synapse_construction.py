#!/usr/bin/env python
'''
An example of constructing synapses.
'''

from brian import *
import time

N=10

P=NeuronGroup(N,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(N,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w')

S[:,:]='i==j'
S.w='2*i'

M=StateMonitor(Q,'v',record=True)

run(40*ms)

for i in range(N):
    plot(M.times/ms,M[i]+i*2,'k')
show()
