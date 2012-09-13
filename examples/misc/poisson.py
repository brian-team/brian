#!/usr/bin/env python
"""
This example demonstrates the PoissonGroup object. Here we have
used a custom function to generate different rates at different
times.

This example also demonstrates a custom SpikeMonitor.
"""
#import brian_no_units # uncomment to run faster
from brian import *

# Rates

r1 = arange(101, 201) * 0.1 * Hz
r2 = arange(1, 101) * 0.1 * Hz

def myrates(t):
    if t < 10 * second:
        return r1
    else:
        return r2
# More compact: myrates=lambda t: (t<10*second and r1) or r2

# Neuron group
P = PoissonGroup(100, myrates)

# Calculation of rates

ns = zeros(len(P))

def ratemonitor(spikes):
    ns[spikes] += 1

Mf = SpikeMonitor(P, function=ratemonitor)
M = SpikeMonitor(P)

# Simulation and plotting

run(10 * second)
print "Rates after 10s:"
print ns / (10 * second)

ns[:] = 0
run(10 * second)
print "Rates after 20s:"
print ns / (10 * second)

raster_plot()
show()
