#!/usr/bin/env python
'''
An example of single-electrode current clamp recording
with bridge compensation (using the electrophysiology library).
'''
from brian import *
from brian.library.electrophysiology import *

taum = 20 * ms        # membrane time constant
gl = 1. / (50 * Mohm)   # leak conductance
Cm = taum * gl        # membrane capacitance
Re = 50 * Mohm        # electrode resistance
Ce = 0.5 * ms / Re      # electrode capacitance

eqs = Equations('''
dvm/dt=(-gl*vm+i_inj)/Cm : volt
Rbridge:ohm # bridge resistance
I:amp # command current
''')
eqs += current_clamp(i_cmd='I', Re=Re, Ce=Ce, bridge='Rbridge')
setup = NeuronGroup(1, model=eqs)
soma = StateMonitor(setup, 'vm', record=True)
recording = StateMonitor(setup, 'v_rec', record=True)

# No compensation
run(50 * ms)
setup.I = .5 * nA
run(100 * ms)
setup.I = 0 * nA
run(50 * ms)

# Full compensation
setup.Rbridge = Re
run(50 * ms)
setup.I = .5 * nA
run(100 * ms)
setup.I = 0 * nA
run(50 * ms)

plot(recording.times / ms, recording[0] / mV, 'b')
plot(soma.times / ms, soma[0] / mV, 'r')
show()
