#!/usr/bin/env python
'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This simulation takes a long time!

Original time: 278 s
with DelayConnection: 478 s
New time: 416 s
'''
from brian import *
from time import time

N = 1000
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
S = Synapses(input, neurons,
             model='''w:1
                      dApre/dt=-Apre/taupre : 1 (event-driven)
                      dApost/dt=-Apost/taupost : 1 (event-driven)''',
             pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax)''',
             post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''')
neurons.v = vr
S[:,:]=True
S.w='rand()*gmax'

rate = PopulationRateMonitor(neurons)

start_time = time()
run(100 * second, report='text')
print "Simulation time:", time() - start_time

subplot(311)
plot(rate.times / second, rate.smooth_rate(100 * ms))
subplot(312)
plot(S.w[:] / gmax, '.')
subplot(313)
hist(S.w[:] / gmax, 20)
show()
