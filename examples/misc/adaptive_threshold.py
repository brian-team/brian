#!/usr/bin/env python
'''
A model with adaptive threshold (increases with each spike)
'''
from brian import *

eqs = '''
dv/dt = -v/(10*ms) : volt
dvt/dt = (10*mV-vt)/(15*ms) : volt
'''

reset = '''
v=0*mV
vt+=3*mV
'''

IF = NeuronGroup(1, model=eqs, reset=reset, threshold='v>vt')
IF.rest()
PG = PoissonGroup(1, 500 * Hz)

C = Connection(PG, IF, 'v', weight=3 * mV)

Mv = StateMonitor(IF, 'v', record=True)
Mvt = StateMonitor(IF, 'vt', record=True)

run(100 * ms)

plot(Mv.times / ms, Mv[0] / mV)
plot(Mvt.times / ms, Mvt[0] / mV)

show()
