#!/usr/bin/env python
'''
Random network with external noise and transmission delays
'''
from brian import *
tau = 10 * ms
sigma = 5 * mV
eqs = 'dv/dt = -v/tau+sigma*xi/tau**.5 : volt'
P = NeuronGroup(4000, model=eqs, threshold=10 * mV, reset=0 * mV, \
              refractory=5 * ms)
P.v = -60 * mV
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
C = Connection(P, P, 'v', delay=2 * ms)
C.connect_random(Pe, P, 0.05, weight=.7 * mV)
C.connect_random(Pi, P, 0.05, weight= -2.8 * mV)
M = SpikeMonitor(P, True)
run(1 * second)
print 'Mean rate =', M.nspikes / 4000. / second
raster_plot(M)
show()
