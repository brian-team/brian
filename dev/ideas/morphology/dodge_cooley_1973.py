"""
Dodge and Cooley 1973. Model of a motoneuron.
"""
from brian import *
from morphology import *
from spatialneuron import *

defaultclock.dt=0.01*ms

morpho=Cylinder(length=1000*um, diameter=10*um, n=100, type='axon')

El = 0 * mV
ENa = 115*mV
EK = -5 * mV
gl = 1 * msiemens / cm ** 2
gNa = 600*gl
gK = 100*gl

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+I : amp/cm**2 # distributed transmembrane current
I:amp/cm**2 # applied current
dm/dt=alpham*(1-m)-betam*m : 1
dn/dt=alphan*(1-n)-betan*n : 1
dh/dt=alphah*(1-h)-betah*h : 1
alpham=(0.4/mV)*(25*mV-v)/(exp((25*mV-v)/(5*mV))-1)/ms : Hz
betam=(0.4/mV)*(v-55*mV)/(exp((v-55*mV)/(5*mV))-1)/ms : Hz
alphah=0.28*exp((10*mV-v)/(20*mV))/ms : Hz
betah=4/(exp((40*mV-v)/(10*mV))+1)/ms : Hz
alphan=(0.2/mV)*(20*mV-v)/(exp((20*mV-v)/(10*mV))-1)/ms : Hz
betan=0.25*exp((10*mV-v)/(80*mV))/ms : Hz
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=El
neuron.h=1
neuron.I=0*amp/cm**2
M=StateMonitor(neuron,'v',record=True)

print 'taum=',neuron.Cm/gl

run(50*ms)
neuron.I[0]=100 * nA/neuron.area[0] # current injection at one end
run(200*ms)

for i in range(10):
    plot(M.times/ms,M[i*10]/mV)
show()
