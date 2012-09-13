#!/usr/bin/env python
'''
Adapted from
Theory of Arachnid Prey Localization
W. Sturzl, R. Kempter, and J. L. van Hemmen
PRL 2000

Poisson inputs are replaced by integrate-and-fire neurons

Romain Brette
'''
from brian import *

# Parameters
degree = 2 * pi / 360.
duration = 500 * ms
R = 2.5 * cm # radius of scorpion
vr = 50 * meter / second # Rayleigh wave speed
phi = 144 * degree # angle of prey
A = 250 * Hz
deltaI = .7 * ms # inhibitory delay
gamma = (22.5 + 45 * arange(8)) * degree # leg angle
delay = R / vr * (1 - cos(phi - gamma))  # wave delay

# Wave (vector w)
t = arange(int(duration / defaultclock.dt) + 1) * defaultclock.dt
Dtot = 0.
w = 0.
for f in range(150, 451):
    D = exp(-(f - 300) ** 2 / (2 * (50 ** 2)))
    xi = 2 * pi * rand()
    w += 100 * D * cos(2 * pi * f * t + xi)
    Dtot += D
w = .01 * w / Dtot

# Rates from the wave
def rates(t):
    return w[array(t / defaultclock.dt, dtype=int)]

# Leg mechanical receptors
tau_legs = 1 * ms
sigma = .01
eqs_legs = """
dv/dt=(1+rates(t-d)-v)/tau_legs+sigma*(2./tau_legs)**.5*xi:1
d : second
"""
legs = NeuronGroup(8, model=eqs_legs, threshold=1, reset=0, refractory=1 * ms)
legs.d = delay
spikes_legs = SpikeCounter(legs)

# Command neurons
tau = 1 * ms
taus = 1 * ms
wex = 7
winh = -2
eqs_neuron = '''
dv/dt=(x-v)/tau : 1
dx/dt=(y-x)/taus : 1 # alpha currents
dy/dt=-y/taus : 1
'''
neurons = NeuronGroup(8, model=eqs_neuron, threshold=1, reset=0)
synapses_ex = IdentityConnection(legs, neurons, 'y', weight=wex)
synapses_inh = Connection(legs, neurons, 'y', delay=deltaI)
for i in range(8):
    synapses_inh[i, (4 + i - 1) % 8] = winh
    synapses_inh[i, (4 + i) % 8] = winh
    synapses_inh[i, (4 + i + 1) % 8] = winh
spikes = SpikeCounter(neurons)

run(duration)
nspikes = spikes.count
x = sum(nspikes * exp(gamma * 1j))
print "Angle (deg):", arctan(imag(x) / real(x)) / degree
polar(concatenate((gamma, [gamma[0] + 2 * pi])), concatenate((nspikes, [nspikes[0]])) / duration)
show()
