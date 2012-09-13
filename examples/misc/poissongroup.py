#!/usr/bin/env python
'''
Poisson input to an IF model
'''
from brian import *

PG = PoissonGroup(1, lambda t:200 * Hz * (1 + cos(2 * pi * t * 50 * Hz)))
IF = NeuronGroup(1, model='dv/dt=-v/(10*ms) : volt', reset=0 * volt, threshold=10 * mV)

C = Connection(PG, IF, 'v', weight=3 * mV)

MS = SpikeMonitor(PG, True)
Mv = StateMonitor(IF, 'v', record=True)
rates = StateMonitor(PG, 'rate', record=True)

run(100 * ms)

subplot(211)
plot(rates.times / ms, rates[0] / Hz)
subplot(212)
plot(Mv.times / ms, Mv[0] / mV)

show()
