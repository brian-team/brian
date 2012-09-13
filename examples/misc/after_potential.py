#!/usr/bin/env python
"""
A model with depolarizing after-potential.
"""
from brian import *

v0=20.5*mV
eqs = '''
dv/dt = (v0-v)/(30*ms) : volt # the membrane equation
dAP/dt = -AP/(3*ms) : volt # the after-potential
vm = v+AP : volt # total membrane potential
'''
IF = NeuronGroup(1, model=eqs, threshold='vm>20*mV',
                 reset='v=0*mV; AP=10*mV')
Mv = StateMonitor(IF, 'vm', record=True)

run(500 * ms)
plot(Mv.times / ms, Mv[0] / mV)
show()
