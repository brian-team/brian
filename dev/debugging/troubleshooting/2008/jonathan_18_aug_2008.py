from brian import *
from customrefractoriness import *

#----Simulation parameters
cl=Clock(dt=0.1*ms)
period=5*ms   #period of time over which the slope of depolarization is computed   
n=int(period/cl.dt)
N=1
k=16*mV/ms

#----Fixed parameters
ENa=60*mV
El=-70*mV
tau=5*ms
tauh=5*ms

#----Input parameters
mu=5*mV
sigma=2*15*mV
tauc=10*ms

#----Na channels parameters
va=-28.69*mV
ka=6.77*mV
vi=-71.87*mV
ki=7.65*mV
r=10.8

eqs=Equations("""
# Sodium channel model
Pa=1/(1+exp(-(v-va)/ka)) : 1
Pi=1/(1+exp(-(v-vi)/ki)) : 1

# Membrane equation
dv/dt=(-r*Pa*h*(v-ENa)-(v-El)+I)/tau : volt
dh/dt=((1-Pi)-h)/tauh : 1

# Input
dI/dt=(mu-I)/tauc+sigma*(.5*tauc)**-.5*xi : volt
""")

def myreset(P, spikes):
    P.v[spikes]=El
    P.h[spikes]=1

#neuron=NeuronGroup(N,model=eqs,threshold=0*mV,reset=myreset,refractory=50*ms)
neuron=NeuronGroup(N, model=eqs, threshold=0*mV, reset=CustomRefractoriness(myreset, 50*ms))

#----Initial conditions
neuron.v=El
neuron.h=1

#----Record variables
Mv=StateMonitor(neuron, 'v', record=0)
Mh=StateMonitor(neuron, 'h', record=0)
Msp=SpikeMonitor(neuron)

run(150*ms)

#--Fig2:Parameters time course---------------
figure()
subplot(211)
plot(Mv.times/ms, Mv[0]/mV)
subplot(212)
plot(Mh.times/ms, Mh[0])

show()
