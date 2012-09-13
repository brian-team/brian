#!/usr/bin/env python
"""
A ring of integrate-and-fire neurons.
"""
from brian import *

tau = 10 * ms
v0 = 11 * mV
N = 20
w = 1 * mV

ring = NeuronGroup(N, model='dv/dt=(v0-v)/tau : volt', threshold=10 * mV, reset=0 * mV)

W = Connection(ring, ring, 'v')
for i in range(N):
    W[i, (i + 1) % N] = w

ring.v = rand(N) * 10 * mV

S = SpikeMonitor(ring)

run(300 * ms)

raster_plot(S)
show()
