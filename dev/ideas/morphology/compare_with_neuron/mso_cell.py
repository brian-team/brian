'''
A pseudo MSO neuron, with two dendrites and one axon (fake geometry).
'''
from brian import *
from brian.experimental.morphology import *

defaultclock.dt=0.1*ms

# Morphology
morpho=Soma(30*um)
morpho.axon=Cylinder(diameter=1*um,length=300*um,n=100)
morpho.L=Cylinder(diameter=3*um,length=100*um,n=50)
morpho.R=Cylinder(diameter=3*um,length=150*um,n=50)

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
neuron.gL=1e-4*siemens/cm**2

# Monitors
mon=StateMonitor(neuron,'v',record=[0,50])

run(1*ms)
neuron.I[0]=0.2*nA/neuron.area[0]
#neuron.changed=True
run(50*ms)
neuron.I=0*amp
#neuron.changed=True
run(495*ms,report='text')

# Load Neuron data
file=r'D:\My Dropbox\LocalEclipseWorkspace\Neuron\mso_cell.dat'
x,y,z=read_neuron_dat(file)

subplot(211)
plot(x,y,'b')
plot(mon.times/ms,mon[0]/mV,'r')
subplot(212)
plot(x,z,'b')
plot(mon.times/ms,mon[50]/mV,'r')
show()
