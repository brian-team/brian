from brian import *
from time import time

N = 1 #number of neurons

# Parameters
C=281*pF
gL=30*nS
taum=C/gL
EL=-70.6*mV
VT=-50.4*mV
DeltaT=2*mV
Vcut=VT+5*DeltaT

b=0*nA
a=3*gL
tauw=2*taum
Vr=-70.6*mV

#For the sinusoidal stim current
ampl = 1.0*nA#*nS*mV
f=10/second#*Hz

eqs="""
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I-w + ampl*sin(2*pi*f*t))/C : volt
dw/dt=(a*(vm-EL)-w)/tauw : amp
I : amp
"""

neuron=NeuronGroup(N,model=eqs,threshold=Vcut,reset="vm=Vr;w+=b",freeze=True)
neuron.vm=EL
trace=StateMonitor(neuron,'vm',record=0)
spikes=SpikeMonitor(neuron)

run(100*ms)

plot(trace.times/ms,trace[0]/mV)
show()
