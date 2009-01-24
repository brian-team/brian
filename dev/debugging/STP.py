'''
Example with Tsodyks STP model
Neurons with regular inputs and depressing synapses
'''
from brian import *

U_SE=.67
tau_e=3*ms
taum=50*ms
tau_rec=800*ms
A_SE=250*pA
Rm=100*Mohm
N=10

eqs='''
dx/dt=rate : 1
rate : Hz
'''

input=NeuronGroup(N,model=eqs,threshold=1.,reset=0)
input.rate=linspace(5*Hz,30*Hz,N)

eqs_neuron='''
dv/dt=(Rm*i-v)/taum:volt
di/dt=-i/tau_e:amp
'''
neuron=NeuronGroup(N,model=eqs_neuron)

C=Connection(input,neuron,'i')
C.connect_one_to_one(input,neuron,A_SE*U_SE)
stp=STP(C,taud=tau_rec,tauf=tau_rec,U=U_SE)
trace=StateMonitor(neuron,'v',record=[0,N-1])

run(1000*ms)
subplot(211)
plot(trace.times/ms,trace[0]/mV)
title('Vm')
subplot(212)
plot(trace.times/ms,trace[N-1]/mV)
title('Vm')
show()
