#!/usr/bin/env python
'''
An example of single-electrode current clamp recording
with discontinuous current clamp (using the electrophysiology library).
'''
from brian import *
from brian.library.electrophysiology import *

defaultclock.dt = 0.01 * ms

taum = 20 * ms        # membrane time constant
gl = 1. / (50 * Mohm)   # leak conductance
Cm = taum * gl        # membrane capacitance
Re = 50 * Mohm        # electrode resistance
Ce = 0.1 * ms / Re      # electrode capacitance

eqs = Equations('''
dvm/dt=(-gl*vm+i_inj)/Cm : volt
Rbridge:ohm # bridge resistance
I:amp # command current
''')
eqs += current_clamp(i_cmd='I', Re=Re, Ce=Ce)
setup = NeuronGroup(1, model=eqs)
ampli = DCC(setup, 'v_rec', 'I', 1 * kHz)
soma = StateMonitor(setup, 'vm', record=True)
recording = StateMonitor(setup, 'v_rec', record=True)
DCCrecording = StateMonitor(ampli, 'record', record=True)

# No compensation
run(50 * ms)
ampli.command = .5 * nA
run(100 * ms)
ampli.command = 0 * nA
run(50 * ms)

ampli.set_frequency(2 * kHz)
ampli.command = .5 * nA
run(100 * ms)
ampli.command = 0 * nA
run(50 * ms)

plot(recording.times / ms, recording[0] / mV, 'b')
plot(DCCrecording.times / ms, DCCrecording[0] / mV, 'k')
plot(soma.times / ms, soma[0] / mV, 'r')
show()
