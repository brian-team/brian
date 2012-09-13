#!/usr/bin/env python
'''
Input-Frequency curve of a neuron (cortical RS type)
Network: 1000 unconnected integrate-and-fire neurons (Brette-Gerstner)
with an input parameter I.
The input is set differently for each neuron.
Spikes are sent to a 'neuron' group with the same size and variable n,
which has the role of a spike counter.
'''
from brian import *
from brian.library.IF import *

N = 1000
eqs = Brette_Gerstner() + Current('I:amp')
print eqs
group = NeuronGroup(N, model=eqs, threshold= -20 * mV, reset=AdaptiveReset())
group.vm = -70 * mV
group.I = linspace(0 * nA, 1 * nA, N)

counter = NeuronGroup(N, model='n:1')
C = IdentityConnection(group, counter, 'n')

i = N * 8 / 10
trace = StateMonitor(group, 'vm', record=i)

duration = 5 * second
run(duration)
subplot(211)
plot(group.I / nA, counter.n / duration)
xlabel('I (nA)')
ylabel('Firing rate (Hz)')
subplot(212)
plot(trace.times / ms, trace[i] / mV)
xlabel('Time (ms)')
ylabel('Vm (mV)')
show()
