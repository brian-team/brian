#!/usr/bin/env python
'''
A rate model
'''
from brian import *

N = 50000
tau = 20 * ms
I = 10 * Hz
eqs = '''
dv/dt=(I-v)/tau : Hz # note the unit here: this is the output rate
'''
group = NeuronGroup(N, eqs, threshold=PoissonThreshold())
S = PopulationRateMonitor(group, bin=1 * ms)

run(100 * ms)

plot(S.rate)
show()
