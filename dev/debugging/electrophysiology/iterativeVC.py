# Iterative voltage-clamp
# R. Brette 2008
from brian import *
#from brian.library.amplifier import *
#from brian.library.electrodes import *

C=200*pF
gl=C/(20*ms)
E=20*mV # holding potential
g=30*nS # initial gain

eqs='''
dv/dt=(-gl*v+I)/C : volt
I : amp
'''

neuron=NeuronGroup(1,model=eqs)
mon=StateMonitor(neuron,'v',record=0)

j=0
current=zeros(500)*amp
@network_operation
def inject():
    global j
    if j<500:
        neuron.I=current[j]
        j=j+1
    else:
        neuron.I=0*nA

err=1e15*mV**2
for i in range(100):
    j=0
    run(50*ms)
    
    v=mon[0][-500:]
    err=sum((E-v)**2) # clamp error
    print err
    #current=g*(E-v)
    #g=1.2*g
    current+=g*(E-v)
    # use best combination of last 2? + new independent component
    #g=2.*g

    run(100*ms) # rest

plot(mon.times/ms,mon[0]/mV)
show()
