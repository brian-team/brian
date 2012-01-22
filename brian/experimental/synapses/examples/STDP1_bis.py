'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)

This simulation takes a long time!

Original time: 278 s
with DelayConnection: 478 s
New time: 479 s
    with precomputed offsets: 444 s
    with zero delays: 418 s
    with weave: 479 s (??)
'''
from brian import *
from brian.experimental.synapses import *
from time import time
set_global_preferences(useweave=False)

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
dA_post *= gmax
dA_pre *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
S = Synapses(input, neurons,
             model='''w:1
                      dA_pre/dt=-A_pre/tau_pre : 1 (event-driven)
                      dA_post/dt=-A_post/tau_post : 1 (event-driven)''',
             pre='''ge+=w
                    A_pre+=dA_pre
                    w=clip(w+A_post,0,gmax)''',
             post='''A_post+=dA_post
                     w=clip(w+A_pre,0,gmax)''')
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
