#!/usr/bin/env python
'''
An auditory filterbank implemented with Poisson neurons

The input sound has a missing fundamental (only harmonics 2 and 3)
'''
from brian import *

defaultclock.dt = .01 * ms

N = 1500
tau = 1 * ms # Decay time constant of filters = 2*tau
freq = linspace(100 * Hz, 2000 * Hz, N) # characteristic frequencies
f_stimulus = 500 * Hz # stimulus frequency
gain = 500 * Hz

eqs = '''
dv/dt=(-a*w-v+I)/tau : Hz
dw/dt=(v-w)/tau : Hz # e.g. linearized potassium channel with conductance a
a : 1
I = gain*(sin(4*pi*f_stimulus*t)+sin(6*pi*f_stimulus*t)) : Hz
'''

neurones = NeuronGroup(N, model=eqs, threshold=PoissonThreshold())
neurones.a = (2 * pi * freq * tau) ** 2

spikes = SpikeMonitor(neurones)
counter = SpikeCounter(neurones)
run(100 * ms)

subplot(121)
CF = array([freq[i] for i, _ in spikes.spikes])
timings = array([t for _, t in spikes.spikes])
plot(timings / ms, CF, '.')
xlabel('Time (ms)')
ylabel('Characteristic frequency (Hz)')
subplot(122)
plot(counter.count / (300 * ms), freq)
xlabel('Firing rate (Hz)')
show()
