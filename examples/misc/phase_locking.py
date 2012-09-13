#!/usr/bin/env python
'''
Phase locking of IF neurons to a periodic input
'''
from brian import *

tau = 20 * ms
N = 100
b = 1.2 # constant current mean, the modulation varies
f = 10 * Hz

eqs = '''
dv/dt=(-v+a*sin(2*pi*f*t)+b)/tau : 1
a : 1
'''

neurons = NeuronGroup(N, model=eqs, threshold=1, reset=0)
neurons.v = rand(N)
neurons.a = linspace(.05, 0.75, N)
S = SpikeMonitor(neurons)
trace = StateMonitor(neurons, 'v', record=50)

run(1000 * ms)
subplot(211)
raster_plot(S)
subplot(212)
plot(trace.times / ms, trace[50])
show()
