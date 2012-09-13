#!/usr/bin/env python
'''
Spike-based adaptation of Licklider's model of pitch processing (autocorrelation with
delay lines) with phase locking.

Romain Brette
'''
from brian import *

defaultclock.dt = .02 * ms

# Ear and sound
max_delay = 20 * ms # 50 Hz
tau_ear = 1 * ms
sigma_ear = .1
eqs_ear = '''
dx/dt=(sound-x)/tau_ear+sigma_ear*(2./tau_ear)**.5*xi : 1
sound=5*sin(2*pi*frequency*t)**3 : 1 # nonlinear distorsion
#sound=5*(sin(4*pi*frequency*t)+.5*sin(6*pi*frequency*t)) : 1 # missing fundamental
frequency=(200+200*t*Hz)*Hz : Hz # increasing pitch
'''
receptors = NeuronGroup(2, model=eqs_ear, threshold=1, reset=0, refractory=2 * ms)
traces = StateMonitor(receptors, 'x', record=True)
sound = StateMonitor(receptors, 'sound', record=0)

# Coincidence detectors
min_freq = 50 * Hz
max_freq = 1000 * Hz
N = 300
tau = 1 * ms
sigma = .1
eqs_neurons = '''
dv/dt=-v/tau+sigma*(2./tau)**.5*xi : 1
'''
neurons = NeuronGroup(N, model=eqs_neurons, threshold=1, reset=0)
synapses = Connection(receptors, neurons, 'v', structure='dense', max_delay=1.1 * max_delay, delay=True)
synapses.connect_full(receptors, neurons, weight=.5)
synapses.delay[1, :] = 1. / exp(linspace(log(min_freq / Hz), log(max_freq / Hz), N))
spikes = SpikeMonitor(neurons)

run(500 * ms)
raster_plot(spikes)
ylabel('Frequency')
yticks([0, 99, 199, 299], array(1. / synapses.delay.todense()[1, [0, 99, 199, 299]], dtype=int))
show()
