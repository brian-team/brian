from brian import *

from brian.units import check_units,siemens,volt
from brian.membrane_equations import Current

def leak_current_int(gl,El,current_name=None):
     return Current('I=gl*(El-vm) : amp',gl=gl,El=El,I=current_name)

def K_current_int(gmax,EK,current_name=None):
     return Current('''
     I=gmax*n**4*(EK-vm) : amp
     dn/dt=alphan*(1-n)-betan*n : 1
     alphan=.05*(vm+34.*mV)/(1-exp(-(vm+34.*mV)/(10.*mV)))/mV/ms : Hz
     betan=.625*exp(-(vm+44.*mV)/(80.*mV))/ms : Hz
     ''',gmax=gmax,EK=EK,I=current_name)

def Na_current_int(gmax,ENa,current_name=None):
     return Current('''
     I=gmax/((1+betam/alpham)**3)*h*(ENa-vm) : amp
     dh/dt=alphah*(1-h)-betah*h : 1
     alpham=.5*(vm+35.*mV)/(1-exp(-(vm+35.*mV)/(10.*mV)))/mV/ms : Hz
     betam=20.*exp(-(vm+60.*mV)/(18.*mV))/ms : Hz
     alphah=.35*exp(-(vm+58.*mV)/(20.*mV))/ms : Hz
     betah=5./(1.+exp(-(vm+28.*mV)/(10.*mV)))/ms : Hz
     ''',gmax=gmax,ENa=ENa,I=current_name)

c=Clock(dt=.01*ms) # more precise
El=-65*mV
EK=-90*mV
ENa=55*mV
eqs=MembraneEquation(0.01*uF)+leak_current_int(.003*msiemens,El,current_name='Il')
eqs+=K_current_int(0.36*msiemens,EK,current_name='IK')
eqs+=Na_current_int(1.20*msiemens,ENa)
eqs+=Current(I='Iapp:amp')

neuron=NeuronGroup(1,eqs)

neuron.h=1.
neuron.n=.1
neuron.vm=-64*mV

trace=StateMonitor(neuron,'vm',record=True)

run(100*ms)
neuron.Iapp=0.

run(100*ms)
print trace[0]/mV
plot(trace.times/ms,trace[0]/mV)
show()
