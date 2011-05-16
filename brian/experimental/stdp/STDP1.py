'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This simulation takes a long time!
'''
from brian import *
from time import time
from eventbased_stdp import *

#log_level_debug() # uncomment to see the STDP equations

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
dA_pre = .01
dA_post = -dA_pre * tau_pre / tau_post * 1.05

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
synapses = Connection(input, neurons, 'ge', weight=rand(len(input), len(neurons)) * gmax)#, structure='dense')
neurons.v = vr

## Explicit STDP rule
eqs_stdp = '''
A_pre : 1
A_post : 1
last_t : second # last update
'''
dA_post *= gmax
dA_pre *= gmax
pre = """
A_pre=A_pre*exp(-(t-last_t)/tau_pre)+dA_pre
A_post=A_post*exp(-(t-last_t)/tau_post)
w+=A_post
last_t = t
"""
post = """
A_pre=A_pre*exp(-(t-last_t)/tau_pre)
A_post=A_post*exp(-(t-last_t)/tau_post)+dA_post
w+=A_pre
last_t = t
"""
stdp = EventBasedSTDP(synapses, eqs=eqs_stdp, pre=pre,post=post, wmax=gmax)

rate = PopulationRateMonitor(neurons)

start_time = time()

run(100 * second, report='text')
print "Simulation time:", time() - start_time

figure()
subplot(311)
plot(rate.times / second, rate.smooth_rate(30 * ms))
subplot(312)
plot(synapses.W.todense() / gmax, '.')
subplot(313)
hist(synapses.W.todense() / gmax, 20)
show()
