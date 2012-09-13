#!/usr/bin/env python
'''
Voltage-clamp experiment (SEVC)
'''
from brian import *
from brian.library.electrophysiology import *

defaultclock.dt = .01 * ms

taum = 20 * ms        # membrane time constant
gl = 1. / (50 * Mohm)   # leak conductance
Cm = taum * gl        # membrane capacitance
Re = 50 * Mohm        # electrode resistance
Ce = 0.1 * ms / Re      # electrode capacitance

eqs = Equations('''
dvm/dt=(-gl*vm+i_inj)/Cm : volt
I:amp
''')
eqs += current_clamp(i_cmd='I', Re=Re, Ce=Ce)
setup = NeuronGroup(1, model=eqs)
ampli = SEVC(setup, 'v_rec', 'I', 1 * kHz, gain=250 * nS, gain2=50 * nS / ms)
recording = StateMonitor(ampli, 'record', record=True)
soma = StateMonitor(setup, 'vm', record=True)

ampli.command = 20 * mV
run(200 * ms)
figure()
plot(recording.times / ms, recording[0] / nA, 'k')
figure()
plot(soma.times / ms, soma[0] / mV, 'b')
show()
