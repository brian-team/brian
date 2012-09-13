#!/usr/bin/env python
'''
A very simple example Brian script to show how to implement
a leaky integrate and fire model. In this example, we also
drive the single leaky integrate and fire neuron with
regularly spaced spikes from the :class:`SpikeGeneratorGroup`.
'''
from brian import *

tau = 10 * ms
Vr = -70 * mV
Vt = -55 * mV

G = NeuronGroup(1, model='dV/dt = -(V-Vr)/tau : volt', threshold=Vt, reset=Vr)

spikes = [(0, t*second) for t in linspace(10 * ms, 100 * ms, 25)]
input = SpikeGeneratorGroup(1, spikes)

C = Connection(input, G)
C[0, 0] = 5 * mV

M = StateMonitor(G, 'V', record=True)

G.V = Vr
run(100 * ms)
plot(M.times / ms, M[0] / mV)
show()
