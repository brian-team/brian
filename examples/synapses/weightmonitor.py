#!/usr/bin/env python
"""
Monitoring synaptic variables.
STDP example.
"""
from brian import *
from time import time

N = 1000
taum = 10 * ms
tau_pre = 20 * ms
tau_post = tau_pre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * tau_pre / tau_post * 1.05
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
             Apre:1
             Apost:1''',
             pre='''ge+=w
             Apre=Apre*exp((lastupdate-t)/tau_pre)+dApre
             Apost=Apost*exp((lastupdate-t)/tau_post)
             w=clip(w+Apost,0,gmax)''',
             post='''
             Apre=Apre*exp((lastupdate-t)/tau_pre)
             Apost=Apost*exp((lastupdate-t)/tau_post)+dApost
             w=clip(w+Apre,0,gmax)''')
neurons.v = vr
S[:,:]=True
S.w='rand()*gmax'

rate = PopulationRateMonitor(neurons)
M = StateMonitor(S,'w',record=[0,1]) # monitors synapses number 0 and 1

start_time = time()
run(10 * second, report='text')
print "Simulation time:", time() - start_time

figure()
subplot(311)
plot(rate.times / second, rate.smooth_rate(100 * ms))
subplot(312)
plot(S.w[:] / gmax, '.')
subplot(313)
hist(S.w[:] / gmax, 20)
figure()
plot(M.times,M[0]/gmax)
plot(M.times,M[1]/gmax)
show()
