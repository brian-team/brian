'''
Hodgkin-Huxley equations (1952)

Conduction velocity is about 12.5 m/s (is it right?)
'''
from brian import *
from brian.experimental.morphology import *

defaultclock.dt=0.01*ms

morpho=Cylinder(length=10*cm, diameter=2*238*um, n=1000, type='axon')

vshift=-65*mV

El = 10.613* mV +vshift
ENa = 115*mV +vshift
EK = -12 * mV +vshift
gl = 0.3 * msiemens / cm ** 2
gNa = 120 * msiemens / cm ** 2
gK = 36 * msiemens / cm ** 2

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+I : amp/cm**2 # distributed transmembrane current
I:amp/cm**2 # applied current
dm/dt=alpham*(1-m)-betam*m : 1
dn/dt=alphan*(1-n)-betan*n : 1
dh/dt=alphah*(1-h)-betah*h : 1
v0=v-vshift : volt
alpham=(0.1/mV)*(-v0+25*mV)/(exp((-v0+25*mV)/(10*mV))-1)/ms : Hz
betam=4.*exp(-v0/(18*mV))/ms : Hz
alphah=0.07*exp(-v0/(20*mV))/ms : Hz
betah=1./(exp((-v0+30*mV)/(10*mV))+1)/ms : Hz
alphan=(0.01/mV)*(-v0+10*mV)/(exp((-v0+10*mV)/(10*mV))-1)/ms : Hz
betan=0.125*exp(-v0/(80*mV))/ms : Hz
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=35.4 * ohm * cm)
neuron.v=0*mV+vshift
neuron.h=1
neuron.m=0
neuron.n=.5
neuron.I=0*amp/cm**2
M=StateMonitor(neuron,'v',record=True)

run(50*ms)
neuron.I[0]=1 * uA/neuron.area[0] # current injection at one end
run(3*ms)
neuron.I=0*amp/cm**2
run(50*ms)

# Load Neuron data
file=r'D:\My Dropbox\LocalEclipseWorkspace\Neuron\hh.dat'
x,y,z=read_neuron_dat(file)

subplot(211)
plot(x,y,'b')
plot(M.times/ms,M[0]/mV,'r')
subplot(212)
plot(x,z,'b')
plot(M.times/ms,M[500]/mV,'r')
show()
