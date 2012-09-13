#!/usr/bin/env python
'''
Two connected neurons with delays
'''
from brian import *
tau = 10 * ms
w = -1 * mV
v0 = 11 * mV
neurons = NeuronGroup(2, model='dv/dt=(v0-v)/tau : volt', threshold=10 * mV, reset=0 * mV, \
                    max_delay=5 * ms)
neurons.v = rand(2) * 10 * mV
W = Connection(neurons, neurons, 'v', delay=2 * ms)
W[0, 1] = w
W[1, 0] = w
S = StateMonitor(neurons, 'v', record=True)
#mymonitor=SpikeMonitor(neurons[0])
mymonitor = PopulationSpikeCounter(neurons)

run(500 * ms)
plot(S.times / ms, S[0] / mV)
plot(S.times / ms, S[1] / mV)
show()
