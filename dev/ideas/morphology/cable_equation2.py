"""
Cable equation with Na channel.
(Segment with current injection at one end).
"""
from brian import *
from morphology import *
from spatialneuron import *

defaultclock.dt=0.1*ms

morpho=Cylinder(length=1000*um, diameter=1*um, n=100, type='axon')

El = -70 * mV
ENa = 50*mV
gl = 0.02 * msiemens / cm ** 2
gNa = 100*gl

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+gNa*m**3*(ENa-v)+I : amp/cm**2 # distributed transmembrane current
I:amp/cm**2 # applied current
dm/dt=(minf-m)/(0.3*ms) : 1
minf=1./(1+exp(-(v+30*mV)/(6*mV))) : 1
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, refractory=4 * ms, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=El
neuron.I=0*amp/cm**2
neuron.I[0]=.03 * nA/neuron.area[0] # current injection at one end
M=StateMonitor(neuron,'v',record=True)

print 'taum=',neuron.Cm/gl

run(200*ms)

for i in range(10):
    plot(M.times/ms,M[i*10]/mV)
show()
