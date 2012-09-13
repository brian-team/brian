#!/usr/bin/env python
'''
Neurons with gap junctions
'''
from brian import *

N = 10
v0=1.05
tau=10*ms

eqs = '''
dv/dt=(v0-v+Igap)/tau : 1
Igap : 1 # gap junction current
'''

neurons = NeuronGroup(N, model=eqs, threshold=1, reset=0)
neurons.v=linspace(0,1,N)
trace = StateMonitor(neurons, 'v', record=[0, 5])

S=Synapses(neurons,model='''w:1 # gap junction conductance
                            Igap=w*(v_pre-v_post): 1''')
S[:,:]=True
neurons.Igap=S.Igap
S.w=.02

run(500*ms)

plot(trace.times / ms, trace[0])
plot(trace.times / ms, trace[5])
show()
