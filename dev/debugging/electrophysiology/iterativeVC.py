# Iterative voltage-clamp
# R. Brette 2008
from brian import *
#from brian.library.amplifier import *
#from brian.library.electrodes import *

C=200*pF
tau=20*ms
gl=C/tau
R=1./gl
E=20*mV # holding potential
g=150*nS # initial gain

eqs='''
dv/dt=(-gl*v+I)/C : volt
I : amp
'''

neuron=NeuronGroup(1, model=eqs)
mon=StateMonitor(neuron, 'v', record=0)

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

dt=defaultclock.dt
err=1e15*mV**2
prev_err=err
maxcur=2*(C/dt)*E
run(10*ms)
#E=E*(1-exp(-linspace(0*ms,50*ms,500)/(1*ms)))
for i in range(100):
    j=0
    run(50*ms)

    v=mon[0][-500:]
    v+=rand(len(v))*2*mV
    err=sum(E-v) # clamp error
    print sqrt(mean((E-v)**2))/float(mV)
    #print err
    #current=g*(E-v)
    #g=1.2*g
    #current+=g*(E-v) # Park's algorithm
    prev=zeros(len(current))
    prev[1:]=(E-v)[0:-1]
    #current+=C/dt*(E-v-exp(-dt/tau)*prev) # Kawato et al
    current+=g*(E-v-.99*prev)
    current=clip(current,-maxcur, maxcur)
    # use best combination of last 2? + new independent component
    #g=2.*g

    run(100*ms) # rest

plot(mon.times/ms, mon[0]/mV)
show()
