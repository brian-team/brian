''' 
STDP model adapted from Song and Abbott 2001
IN PROGRESS
'''
from brian import *
from time import time

# Parameters

taum=20*ms
tau_post=20*ms
tau_pre=20*ms
Ee=0*mV
vt=-54*mV
vr=-60*mV
El=-74*mV
taue=5*ms
gmax=0.015
dA_pre=gmax*.001
dA_post=-dA_pre*1.06
rate_ext=500*Hz
g_ext=.096
N=200

eqs_neurons='''
dv/dt=(ge*(Ee-v)+El-v)/taum : volt
dge/dt=-ge/taue : 1
'''

input=PoissonGroup(1000,rates=10*Hz)
neurons=NeuronGroup(N,model=eqs_neurons,threshold=vt,reset=vr)
synapses=Connection(input,neurons,'ge',structure='dense')
synapses.connect(input,neurons,rand(len(input),len(neurons))*gmax)
synapses_rec=Connection(neurons,neurons,'ge',structure='dense')
f=lambda i,j:0.5*gmax*exp(-.5*(.005*(i-j))**2)
synapses_rec.connect_full(neurons,neurons,weight=lambda i,j:f(i,j)+f(i+N,j)+f(i,j+N))
neurons.v=vr

stdp_in=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)
stdp_rec=ExponentialSTDP(synapses_rec,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)

rate=PopulationRateMonitor(neurons)

print "Starting simulation..."
start_time=time()
run(2*second)
print "Simulation time:",time()-start_time

subplot(211)
plot(rate.times/ms,rate.smooth_rate(200*ms))
subplot(212)
plot(synapses.W.todense(),'.')
show()
