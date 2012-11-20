"""
The ball-and-stick model

There is a 20% difference in access resistance, compared to Neuron
"""
from brian import *
from brian.experimental.morphology import *

defaultclock.dt=0.025*ms

# Morphology
'''
morpho=Morphology(n=101)
morpho.length[0]=30*um
morpho.diameter[0]=30*um
morpho.length[1:]=3*um
morpho.diameter[1:]=1*um
morpho.set_area()
morpho.set_coordinates()
'''

morpho=Soma(diameter=30*um)
morpho.axon=Cylinder(diameter=1*um,length=300*um,n=100)

# Passive channels
gL=1e-4*siemens/cm**2
EL=-70*mV
eqs='''
Im=gL*(EL-v)+I : amp/cm**2
I : amp/cm**2
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=-65*mV
neuron.I=0*amp/cm**2

# Monitors
mon=StateMonitor(neuron,'v',record=[0,50])

run(1*ms)
neuron.I[0]=0.2*nA/neuron.area[0]
run(50*ms)
neuron.I=0*amp
run(949*ms,report='text')

# Load Neuron data
file=r'C:\My Dropbox\LocalEclipseWorkspace\Neuron\example.dat'
x,y,z=read_neuron_dat(file)

subplot(211)
plot(x,y)
plot(mon.times/ms,mon[0]/mV)
subplot(212)
plot(x,z)
plot(mon.times/ms,mon[50]/mV)
#plot(x,z)
show()
