#!/usr/bin/env python
"""
Mirollo-Strogatz network
"""
from brian import *

tau = 10 * ms
v0 = 11 * mV
N = 20
w = .1 * mV

group = NeuronGroup(N, model='dv/dt=(v0-v)/tau : volt', threshold=10 * mV, reset=0 * mV)

W = Connection(group, group, 'v', weight=w)

group.v = rand(N) * 10 * mV

S = SpikeMonitor(group)

run(300 * ms)

raster_plot(S)
show()
