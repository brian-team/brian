#!/usr/bin/env python
'''
This example basically replicates what the Brian PulsePacket object does,
and then compares to that object.
'''

from brian import *
from random import gauss, shuffle

# Generator for pulse packet
def pulse_packet(t, n, sigma):
    # generate a list of n times with Gaussian distribution, sort them in time, and
    # then randomly assign the neuron numbers to them
    times = [gauss(t, sigma) for i in range(n)]
    times.sort()
    neuron = range(n)
    shuffle(neuron)
    return zip(neuron, times) # returns a list of pairs (i,t)

G1 = SpikeGeneratorGroup(1000, pulse_packet(50 * ms, 1000, 5 * ms))
M1 = SpikeMonitor(G1)
PRM1 = PopulationRateMonitor(G1, bin=1 * ms)

G2 = PulsePacket(50 * ms, 1000, 5 * ms)
M2 = SpikeMonitor(G2)
PRM2 = PopulationRateMonitor(G2, bin=1 * ms)

run(100 * ms)

subplot(221)
raster_plot(M1)
subplot(223)
plot(PRM1.rate)
subplot(222)
raster_plot(M2)
subplot(224)
plot(PRM2.rate)
show()
