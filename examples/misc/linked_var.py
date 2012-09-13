#!/usr/bin/env python
'''
Example showing :func:`linked_var`, connecting two different :class:`NeuronGroup`
variables. Here we show something like a simplified haircell and auditory nerve
fibre model where the hair cells and ANFs are implemented as two separate
:class:`NeuronGroup` objects. The hair cells filter their inputs via a 
differential equation, and then emit graded amounts of neurotransmitter
(variable 'y') to the auditory nerve fibres input current (variable 'I').
'''
from brian import *

N = 5
f = 50 * Hz
a_min = 1.0
a_max = 100.0
tau_haircell = 50 * ms
tau = 10 * ms
duration = 100 * ms

eqs_haircells = '''
input = a*sin(2*pi*f*t) : 1
x = clip(input, 0, Inf)**(1.0/3.0) : 1
a : 1
dy/dt = (x-y)/tau_haircell : 1 
'''

haircells = NeuronGroup(N, eqs_haircells)
haircells.a = linspace(a_min, a_max, N)
M_haircells = MultiStateMonitor(haircells, vars=('input', 'y'), record=True)

eqs_nervefibres = '''
dV/dt = (I-V)/tau : 1
I : 1
'''
nervefibres = NeuronGroup(N, eqs_nervefibres, reset=0, threshold=1)
nervefibres.I = linked_var(haircells, 'y')
M_nervefibres = MultiStateMonitor(nervefibres, record=True)

run(duration)

subplot(221)
M_haircells['input'].plot()
ylabel('haircell.input')
subplot(222)
M_haircells['y'].plot()
ylabel('haircell.y')
subplot(223)
M_nervefibres['I'].plot()
ylabel('nervefibres.I')
subplot(224)
M_nervefibres['V'].plot()
ylabel('nervefibres.V')
show()
