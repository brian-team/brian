#!/usr/bin/env python
"""
Example with short term plasticity,
with event-driven updates defined by differential equations.
"""
from brian import *

tau_e = 3 * ms
taum = 10 * ms
A_SE = 250 * pA
Rm = 100 * Mohm
N = 10

eqs = '''
dx/dt=rate : 1
rate : Hz
'''

input = NeuronGroup(N, model=eqs, threshold=1., reset=0)
input.rate = linspace(5 * Hz, 30 * Hz, N)

eqs_neuron = '''
dv/dt=(Rm*i-v)/taum:volt
di/dt=-i/tau_e:amp
'''
neuron = NeuronGroup(N, model=eqs_neuron)

taud=1*ms
tauf=100*ms
U=.1
#taud=100*ms
#tauf=10*ms
#U=.6
S=Synapses(input,neuron,
           model='''w : 1
                    dx/dt=(1-x)/taud : 1 (event-driven)
                    du/dt=(U-u)/tauf : 1 (event-driven)
                 ''',
           pre='''i+=w*u*x
                  x*=(1-u)
                  u+=U*(1-u)''')
S[:,:]='i==j' # one to one connection
S.w=A_SE
# Initialization of STP variables
S.x = 1
S.u = U

trace = StateMonitor(neuron, 'v', record=[0, N - 1])

run(1000 * ms)
subplot(211)
plot(trace.times / ms, trace[0] / mV)
title('Vm')
subplot(212)
plot(trace.times / ms, trace[N - 1] / mV)
title('Vm')
show()
