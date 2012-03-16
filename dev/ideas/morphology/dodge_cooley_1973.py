"""
Dodge and Cooley 1973. Model of a motoneuron.

Doesn't seem to work: no propagation of APs.
"""
from brian import *
from morphology import *
from spatialneuron import *

defaultclock.dt=0.01*ms

morpho=Morphology(n=47)
# Dendrites
morpho.length[:30]=4500*um/30
morpho.diameter[:30]=30*2*um
# Soma
morpho.length[30:36]=300*um/6
morpho.diameter[30:36]=30*2*um
# Initial segment
morpho.length[36:41]=100*um/5
morpho.diameter[36:41]=5*2*um
# Myelin
morpho.length[41:46]=400*um/5
morpho.diameter[41:46]=8*2*um
# Node
morpho.length[46]=75*um
morpho.diameter[46]=10*2*um

morpho.set_area()
morpho.set_coordinates()

El = 0 * mV
ENa = 115*mV
EK = -5 * mV

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gL*(El-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+I : amp/cm**2 # distributed transmembrane current
I:amp/cm**2 # applied current
gNa:siemens/cm**2
gK:siemens/cm**2
gL:siemens/cm**2
Cm:farad/cm**2
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
neuron.m=0
neuron.n=0
neuron.I=0*amp/cm**2
M=StateMonitor(neuron,'v',record=True)

neuron.gL[0:36] = 0.167 * msiemens / cm ** 2
neuron.gNa[30:36] = 70* msiemens / cm ** 2
neuron.gK[30:36] = 17.5* msiemens / cm ** 2

neuron.gL[36:41] = 1 * msiemens / cm ** 2
neuron.gNa[36:41] = 600* msiemens / cm ** 2
neuron.gK[36:41] = 100* msiemens / cm ** 2

neuron.gL[41:46] = 0.05 * msiemens / cm ** 2
neuron.Cm[41:46] = 0.05 * uF / cm ** 2 # myelin
neuron.gNa[41:46] = 0* msiemens / cm ** 2
neuron.gK[41:46] = 0* msiemens / cm ** 2

neuron.gL[46] = 3 * msiemens / cm ** 2
neuron.gNa[46] = 600* msiemens / cm ** 2
neuron.gK[46] = 100* msiemens / cm ** 2

run(50*ms)
neuron.I[30:36]=200 * nA/neuron.area[30] # current injection at the soma
run(5*ms)
neuron.I[30:36]=0 * nA/neuron.area[30]
run(100*ms,report='text')

for i in [33,38,43,46]:
    plot(M.times/ms,M[i]/mV)
show()
