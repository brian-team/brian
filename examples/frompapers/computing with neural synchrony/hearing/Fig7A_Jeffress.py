#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 7A. Jeffress model, adapted with spiking neuron models.
A sound source (white noise) is moving around the head.
Delay differences between the two ears are used to determine the azimuth of the source.
Delays are mapped to a neural place code using delay lines (each neuron receives input
from both ears, with different delays).
'''
from brian import *

defaultclock.dt = .02 * ms
dt = defaultclock.dt

# Sound
sound = TimedArray(10 * randn(50000)) # white noise

# Ears and sound motion around the head (constant angular speed)
sound_speed = 300 * metre / second
interaural_distance = 20 * cm # big head!
max_delay = interaural_distance / sound_speed
print "Maximum interaural delay:", max_delay
angular_speed = 2 * pi * radian / second # 1 turn/second
tau_ear = 1 * ms
sigma_ear = .05
eqs_ears = '''
dx/dt=(sound(t-delay)-x)/tau_ear+sigma_ear*(2./tau_ear)**.5*xi : 1
delay=distance*sin(theta) : second
distance : second # distance to the centre of the head in time units
dtheta/dt=angular_speed : radian
'''
ears = NeuronGroup(2, model=eqs_ears, threshold=1, reset=0, refractory=2.5 * ms)
ears.distance = [-.5 * max_delay, .5 * max_delay]
traces = StateMonitor(ears, 'x', record=True)

# Coincidence detectors
N = 300
tau = 1 * ms
sigma = .05
eqs_neurons = '''
dv/dt=-v/tau+sigma*(2./tau)**.5*xi : 1
'''
neurons = NeuronGroup(N, model=eqs_neurons, threshold=1, reset=0)
synapses = Connection(ears, neurons, 'v', structure='dense', delay=True, max_delay=1.1 * max_delay)
synapses.connect_full(ears, neurons, weight=.5)
synapses.delay[0, :] = linspace(0 * ms, 1.1 * max_delay, N)
synapses.delay[1, :] = linspace(0 * ms, 1.1 * max_delay, N)[::-1]
spikes = SpikeMonitor(neurons)

run(1000 * ms)
raster_plot(spikes)
show()
