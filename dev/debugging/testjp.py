from brian import *

gl=10*nS
El=-70*mV
E=0*mV
C=200*pF
tau_s=5*ms

eqs='''
dv/dt=(gl*(El-v)+g*(E-v))/C : volt
dg/dt=-g/tau_s : siemens
'''

group=NeuronGroup(1,model=eqs,threshold=-55*mV,reset=-70*mV,refractory=2*ms)
group.v=-70*mV
input=PoissonGroup(1,100*Hz)

C=Connection(input,group,'g')
C[0,0]=2*nS

trace=StateMonitor(group,'v',record=True)
run(100*ms)
plot(trace.times/ms,trace[0]/mV)
show()
