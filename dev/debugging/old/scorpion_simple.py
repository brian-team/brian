'''
Adapted (and simplified) from
Theory of Arachnid Prey Localization
W. Sturzl, R. Kempter, and J. L. van Hemmen
PRL 2000

The membrane potential is normalized
(i.e. 0 = rest, 1 = threshold).
'''
from brian import *

# Parameters
radius = 2.5 * cm
wave_speed = 50 * meter / second
prey_angle = 0
legs_angle = pi / 8 + pi / 4 * arange(8)

wave = lambda t:.2 * sin(2 * pi * 300 * Hz * t) * cos(2 * pi * 25 * Hz * t)

# Leg mechanical receptors
tau = 1 * ms
sigma = .01
eqs_legs = """
dv/dt=(1+wave(t-d)-v)/tau+sigma*(2./tau)**.5*xi:1
d : second # wave delay
"""
legs = NeuronGroup(8, model=eqs_legs, threshold=1, reset=0, refractory=1 * ms)
legs.d = radius / wave_speed * (1 - cos(prey_angle - legs_angle))

# Command neurons
eqs_neuron = '''
dv/dt=(x-v)/tau : 1
dx/dt=-x/tau : 1 # PSPs are alpha functions
'''
neurons = NeuronGroup(8, model=eqs_neuron, threshold=1, reset=0)
synapses_ex = IdentityConnection(legs, neurons, 'x', weight=4)
synapses_inh = Connection(legs, neurons, 'x', delay=.7 * ms)
for i in range(8):
    synapses_inh[i, (4 + i - 1) % 8] = -1
    synapses_inh[i, (4 + i) % 8] = -1
    synapses_inh[i, (4 + i + 1) % 8] = -1
spikes = SpikeCounter(neurons)

run(300 * ms)
polar(legs_angle, spikes.count)
print spikes.count
show()
