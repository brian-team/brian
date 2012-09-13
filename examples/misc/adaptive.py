#!/usr/bin/env python
'''
An adaptive neuron model
'''
from brian import *

PG = PoissonGroup(1, 500 * Hz)
eqs = '''
dv/dt = (-w-v)/(10*ms) : volt # the membrane equation
dw/dt = -w/(30*ms) : volt # the adaptation current
'''
# The adaptation variable increases with each spike
IF = NeuronGroup(1, model=eqs, threshold=20 * mV,
                 reset='''v  = 0*mV
                          w += 3*mV ''')

C = Connection(PG, IF, 'v', weight=3 * mV)

MS = SpikeMonitor(PG, True)
Mv = StateMonitor(IF, 'v', record=True)
Mw = StateMonitor(IF, 'w', record=True)

run(100 * ms)

plot(Mv.times / ms, Mv[0] / mV)
plot(Mw.times / ms, Mw[0] / mV)

show()
