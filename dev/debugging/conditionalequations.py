from brian import *

tau=10*ms
x1=-1*volt
x2=0*volt
model='dv/dt=((v>0)*x1+(v<=0)*x2)/tau:volt'
G=NeuronGroup(1,model,method='Euler')
S=StateMonitor(G,'v',record=0)
G.v=1*volt
run(50*ms)

S.plot()
show()
