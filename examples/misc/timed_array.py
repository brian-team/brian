#!/usr/bin/env python
"""
An example of the :class:`TimedArray` class used for applying input currents
to neurons.
"""
from brian import *

N = 5
duration = 100 * ms
Vr = -60 * mV
Vt = -50 * mV
tau = 10 * ms
Rmin = 1 * Mohm
Rmax = 10 * Mohm
freq = 50 * Hz
k = 10 * nA

eqs = '''
dV/dt = (-(V-Vr)+R*I)/tau : volt
R : ohm
I : amp
'''

G = NeuronGroup(N, eqs, reset='V=Vr', threshold='V>Vt')
G.R = linspace(Rmin, Rmax, N)

t = linspace(0 * second, duration, int(duration / defaultclock.dt))
I = clip(k * sin(2 * pi * freq * t), 0, Inf)
G.I = TimedArray(I)

M = MultiStateMonitor(G, record=True)

run(duration)

subplot(211)
M['I'].plot()
ylabel('I (amp)')
subplot(212)
M['V'].plot()
ylabel('V (volt)')
show()
