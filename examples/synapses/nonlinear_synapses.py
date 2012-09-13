#!/usr/bin/env python
"""
NMDA synapses
"""
from brian import *
import time

a=1/(10*ms)
b=1/(10*ms)
c=1/(10*ms)

input=NeuronGroup(2,model='dv/dt=1/(10*ms):1', threshold=1, reset=0)
neurons = NeuronGroup(1, model="""dv/dt=(gtot-v)/(10*ms) : 1
                                  gtot : 1""")
S=Synapses(input,neurons,
           model='''dg/dt=-a*g+b*x*(1-g) : 1
                    dx/dt=-c*x : 1
                    w : 1 # synaptic weight
                 ''',
           pre='x+=w') # NMDA synapses
neurons.gtot=S.g
S[:,:]=True
S.w=[1.,10.]
input.v=[0.,0.5]

M=StateMonitor(S,'g',record=True)
Mn=StateMonitor(neurons,'v',record=0)

run(100*ms)

subplot(211)
plot(M.times/ms,M[0])
plot(M.times/ms,M[1])
subplot(212)
plot(Mn.times/ms,Mn[0])

show()
