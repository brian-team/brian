#!/usr/bin/env python
'''
Integrate-and-fire neurons with noise
'''
from brian import *

tau = 10 * ms
sigma = .5
N = 100
J = -1
mu = 2

eqs = """
dv/dt=mu/tau+sigma/tau**.5*xi : 1
"""

group = NeuronGroup(N, model=eqs, threshold=1, reset=0)

C = Synapses(group, model='w:1', pre='v+=w')
C[:,:]='j==((i+1)%N)'
C.w=J

S = SpikeMonitor(group)
trace = StateMonitor(group, 'v', record=True)

run(500 * ms)
i, t = S.spikes[-1]

subplot(211)
raster_plot(S)
subplot(212)
plot(trace.times / ms, trace[0])
show()
