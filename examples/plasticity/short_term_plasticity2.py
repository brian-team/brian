#!/usr/bin/env python
'''
Network (CUBA) with short-term synaptic plasticity for excitatory synapses
(Depressing at long timescales, facilitating at short timescales)
'''
from brian import *
from time import time

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(4000, model=eqs, threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + rand(4000) * 10 * mV
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
Ce = Connection(Pe, P, 'ge', weight=1.62 * mV, sparseness=.02)
Ci = Connection(Pi, P, 'gi', weight= -9 * mV, sparseness=.02)
stp = STP(Ce, taud=200 * ms, tauf=20 * ms, U=.2)
M = SpikeMonitor(P)
rate = PopulationRateMonitor(P)
t1 = time()
run(1 * second)
t2 = time()
print "Simulation time:", t2 - t1, "s"
print M.nspikes, "spikes"
subplot(211)
raster_plot(M)
subplot(212)
plot(rate.times / ms, rate.smooth_rate(5 * ms))
show()
