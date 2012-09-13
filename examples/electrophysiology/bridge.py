#!/usr/bin/env python
'''
Bridge experiment (current-clamp)
'''
from brian import *
from brian.library.electrophysiology import *

defaultclock.dt = .01 * ms

#log_level_debug()

taum = 20 * ms
gl = 20 * nS
Cm = taum * gl
Re = 50 * Mohm
Ce = 0.5 * ms / Re
N = 10

eqs = Equations('''
dvm/dt=(-gl*vm+i_inj)/Cm : volt
#Rbridge:ohm
CC:farad
I:amp
''')
eqs += electrode(.6 * Re, Ce)
#eqs+=current_clamp(vm='v_el',i_inj='i_cmd',i_cmd='I',Re=.4*Re,Ce=Ce,
#                   bridge='Rbridge')
eqs += current_clamp(vm='v_el', i_inj='i_cmd', i_cmd='I', Re=.4 * Re, Ce=Ce,
                   bridge=Re, capa_comp='CC')
setup = NeuronGroup(N, model=eqs)
setup.I = 0 * nA
setup.v = 0 * mV
#setup.Rbridge=linspace(0*Mohm,60*Mohm,N)
setup.CC = linspace(0 * Ce, Ce, N)
recording = StateMonitor(setup, 'v_rec', record=True)

run(50 * ms)
setup.I = .5 * nA
run(200 * ms)
setup.I = 0 * nA
run(150 * ms)
for i in range(N):
    plot(recording.times / ms + i * 400, recording[i] / mV, 'k')
show()
