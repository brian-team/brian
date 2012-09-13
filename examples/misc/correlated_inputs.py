#!/usr/bin/env python
'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.
'''
from brian import *

N = 100
#input = HomogeneousCorrelatedSpikeTrains(N, r=10 * Hz, c=0.1, tauc=10 * ms)

c = .2
nu = linspace(1*Hz, 10*Hz, N)
P = c*dot(nu.reshape((N,1)), nu.reshape((1,N)))/mean(nu**2)
tauc = 5*ms

spikes = mixture_process(nu, P, tauc, 1*second)
#spikes = [(i,t*second) for i,t in spikes]
input = SpikeGeneratorGroup(N, spikes)

S = SpikeMonitor(input)
#S2 = PopulationRateMonitor(input)
#M = StateMonitor(input, 'rate', record=0)
run(1000 * ms)

#subplot(211)
raster_plot(S)
#subplot(212)
#plot(S2.times / ms, S2.smooth_rate(5 * ms))
#plot(M.times / ms, M[0] / Hz)
show()
