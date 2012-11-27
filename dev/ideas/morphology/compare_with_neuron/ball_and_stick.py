"""
The ball-and-stick model

* There is a 20% difference in access resistance, compared to Neuron, for soma+axon, if it is a single branch
* Cylinder works perfectly (passive).
* Tapered cylinder works almost perfectly, but not exactly.
"""
from brian import *
from brian.experimental.morphology import *

defaultclock.dt=0.1*ms

# Morphology
morpho=Soma(30*um)
morpho.axon=Cylinder(diameter=1*um,length=300*um,n=100)

'''
diam=linspace(10*um,3*um,1000)
morpho=Morphology(n=500)
morpho.length[:]=0.3*um
morpho.diameter[:]=diam[:500] # tapered axon
morpho.set_area()
morpho.set_coordinates()
morpho2=Morphology(n=500)
morpho2.length[:]=0.3*um
morpho2.diameter[:]=diam[500:] # tapered axon
morpho2.set_area()
morpho2.set_coordinates()
morpho.L=morpho2
'''

#morpho=Cylinder(diameter=30*um,length=30*um,n=1)
#morpho.axon=Cylinder(diameter=1*um,length=300*um,n=100)
#morpho=Cylinder(diameter=1*um,length=300*um,n=100)

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
run(495*ms,report='text')

# Load Neuron data
file=r'D:\My Dropbox\LocalEclipseWorkspace\Neuron\example.dat'
x,y,z=read_neuron_dat(file)

subplot(211)
plot(x,y,'b')
plot(mon.times/ms,mon[0]/mV,'r')
subplot(212)
plot(x,z,'b')
plot(mon.times/ms,mon[50]/mV,'r')
show()
