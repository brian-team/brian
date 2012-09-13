#!/usr/bin/env python
'''
Delayed STDP
'''
from brian import *
import time

N = 1
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -74 * mV
El = -74 * mV
taue = 5 * ms
F = 20 * Hz
dApre = .1
dApost = -dApre * taupre / taupost * 2.

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
S = Synapses(input, neurons,
             model='''w:1
                      dApre/dt=-Apre/taupre : 1 #(event-driven)
                      dApost/dt=-Apost/taupost : 1 #(event-driven)''',
             pre=('ge+=w',
                  '''w=clip(w+Apost,0,inf)
                     Apre+=dApre'''),
             post='''Apost+=dApost
                     w=clip(w+Apre,0,inf)''')
neurons.v = vr
S[:,:]=True
S.w=10
S.delay[1][0,0]=3*ms # delayed trace (try 0 ms to see the difference)

M=StateMonitor(S,'w',record=0)
Mpre=StateMonitor(S,'Apre',record=0)
Mpost=StateMonitor(S,'Apost',record=0)
Mv=StateMonitor(neurons,'v',record=0)

run(10*second,report='text')

subplot(211)
plot(M.times/ms,M[0])
plot(M.times/ms,Mpre[0],'r')
plot(M.times/ms,Mpost[0],'k')
subplot(212)
plot(Mv.times/ms,Mv[0])
show()
