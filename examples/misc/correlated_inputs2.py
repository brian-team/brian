#!/usr/bin/env python
'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.
'''
from brian import *

N = 100
c = .2
nu = linspace(1*Hz, 10*Hz, N)
P = c*dot(nu.reshape((N,1)), nu.reshape((1,N)))/mean(nu**2)
tauc = 5*ms

spikes = mixture_process(nu, P, tauc, 1*second)
input = SpikeGeneratorGroup(N, spikes)

S = SpikeMonitor(input)
run(1000 * ms)

raster_plot(S)
show()
