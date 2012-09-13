#!/usr/bin/env python
"""
Probabilistic synapses

Seems to work.
"""
from brian import *

N=20
tau=5*ms
input=PoissonGroup(2,rates=20*Hz)
neurons=NeuronGroup(N,model='dv/dt=-v/tau : 1')

S=Synapses(input,neurons,model="""w : 1
                                  p : 1 # transmission probability""",
                         pre="v+=w*(rand()<p)")
# Transmission probabilities
S[:,:]=True
S.w=0.5
S.p[0,:]=linspace(0,1,N) # transmission probability between 0 and 1
S.p[1,:]=linspace(0,1,N)[::-1] # reverse order for the second input

M=StateMonitor(neurons,'v',record=True)

run(500*ms)

for i in range(N):
    plot(M.times/ms,M[i]+i,'k')
show()
