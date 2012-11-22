"""
The ball-and-stick model
"""
from brian import *
from brian.experimental.morphology import *
#from brian.experimental.morphology.spatialneuron_remy import *

defaultclock.dt=0.025*ms

# Morphology
#morpho=Cylinder(diameter=30*um,length=30*um,n=1)
#morpho.axon=Cylinder(diameter=1*um,length=300*um,n=100)
#morpho=Cylinder(diameter=1*um,length=300*um,n=100)
morpho=Cylinder(diameter=1*um,length=300*um,n=101)

# Passive channels
gL=1e-4*siemens/cm**2
EL=-70*mV
eqs='''
Im=gL*(EL-v)+I : amp/cm**2
I : amp/cm**2
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=-65*mV

# Monitors
mon=StateMonitor(neuron,'v',record=[0,1,50,100])

#run(1*ms)
#neuron.I[0]=0.2*nA/neuron.area[0]
#neuron.changed=True
#run(50*ms)
#neuron.I=0*amp
#neuron.changed=True
run(100*ms,report='text')

print mon[0]/mV

subplot(211)
plot(mon.times/ms,mon[0]/mV,'r')
plot(mon.times/ms,mon[1]/mV,'b')
plot(mon.times/ms,mon[50]/mV,'k')
plot(mon.times/ms,mon[100]/mV,'g')
show()
