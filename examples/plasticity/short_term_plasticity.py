#!/usr/bin/env python
'''
Example with short term plasticity model
Neurons with regular inputs and depressing synapses
'''
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

C = Connection(input, neuron, 'i')
C.connect_one_to_one(weight=A_SE)
stp = STP(C, taud=1 * ms, tauf=100 * ms, U=.1) # facilitation
#stp=STP(C,taud=100*ms,tauf=10*ms,U=.6) # depression
trace = StateMonitor(neuron, 'v', record=[0, N - 1])

run(1000 * ms)
subplot(211)
plot(trace.times / ms, trace[0] / mV)
title('Vm')
subplot(212)
plot(trace.times / ms, trace[N - 1] / mV)
title('Vm')
show()
