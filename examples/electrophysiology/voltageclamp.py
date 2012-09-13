#!/usr/bin/env python
'''
Voltage-clamp experiment
'''
from brian import *
from brian.library.electrophysiology import *

defaultclock.dt = .01 * ms

taum = 20 * ms
gl = 20 * nS
Cm = taum * gl
Re = 50 * Mohm
Ce = 0.2 * ms / Re
N = 1
Rs = .9 * Re
tauc = Rs * Ce # critical tau_u

eqs = Equations('''
dvm/dt=(-gl*vm+i_inj)/Cm : volt
''')
eqs += electrode(.2 * Re, Ce)
eqs += voltage_clamp(vm='v_el', v_cmd=20 * mV, i_inj='i_cmd', i_rec='ic',
                   Re=.8 * Re, Rs=.9 * Re, tau_u=.2 * ms)
setup = NeuronGroup(N, model=eqs)
setup.v = 0 * mV
recording = StateMonitor(setup, 'ic', record=True)
soma = StateMonitor(setup, 'vm', record=True)

run(200 * ms)
figure()
plot(recording.times / ms, recording[0] / nA, 'k')
figure()
plot(soma.times / ms, soma[0] / mV, 'b')
show()
