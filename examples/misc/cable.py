#!/usr/bin/env python
'''
Dendrite with 100 compartments
'''
from brian import *
from brian.compartments import *
from brian.library.ionic_currents import *

length = 1 * mm
nseg = 100
dx = length / nseg
Cm = 1 * uF / cm ** 2
gl = 0.02 * msiemens / cm ** 2
diam = 1 * um
area = pi * diam * dx
El = 0 * mV
Ri = 100 * ohm * cm
ra = Ri * 4 / (pi * diam ** 2)

print "Time constant =", Cm / gl
print "Space constant =", .5 * (diam / (gl * Ri)) ** .5

segments = {}
for i in range(nseg):
    segments[i] = MembraneEquation(Cm * area) + leak_current(gl * area, El)

segments[0] += Current('I:nA')

cable = Compartments(segments)
for i in range(nseg - 1):
    cable.connect(i, i + 1, ra * dx)

neuron = NeuronGroup(1, model=cable)
#neuron.vm_0=10*mV
neuron.I_0 = .05 * nA

trace = []
for i in range(10):
    trace.append(StateMonitor(neuron, 'vm_' + str(10 * i), record=True))

run(200 * ms)

for i in range(10):
    plot(trace[i].times / ms, trace[i][0] / mV)
show()
