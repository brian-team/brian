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

eqs_poisson='''
rate : Hz
dA_pre/dt=-A_pre/tau_pre : 1
'''

eqs_neurons='''
dv/dt=(ge*(Ee-v)+El-v)/taum : volt
dge/dt=-ge/taue : 1
dA_post/dt=-A_post/tau_post : 1
dA_pre/dt=-A_pre/tau_pre : 1
'''

def poisson_reset(P,spikes):
    for i in spikes:
        synapses[i,:]=clip(synapses[i,:]+neurons.A_post_,0,gmax)
    P.A_pre_[spikes]+=dA_pre

def neurons_reset(P,spikes):
    P.v_[spikes]=vr
    for i in spikes:
        synapses[:,i]=clip(synapses[:,i]+input.A_pre_,0,gmax)
        synapses_rec[i,:]=clip(synapses_rec[:,i]+P.A_post_,0,gmax)
        synapses_rec[:,i]=clip(synapses_rec[:,i]+P.A_pre_,0,gmax)
    P.A_pre_[spikes]+=dA_pre
    P.A_post_[spikes]+=dA_post

input=NeuronGroup(1000,model=eqs_poisson,threshold=PoissonThreshold(),reset=poisson_reset)
neurons=NeuronGroup(N,model=eqs_neurons,threshold=vt,reset=neurons_reset)
synapses=Connection(input,neurons,'ge',structure='dense')
synapses.connect(input,neurons,rand(len(input),len(neurons))*gmax)
synapses_rec=Connection(neurons,neurons,'ge',structure='dense')
f=lambda i,j:0.5*gmax*exp(-.5*(.005*(i-j))**2)
synapses_rec.connect_full(neurons,neurons,weight=lambda i,j:f(i,j)+f(i+N,j)+f(i,j+N))
neurons.v=vr
input.rate=10*Hz

rate=PopulationRateMonitor(neurons)

print "Starting simulation..."
start_time=time()
run(500*second)
print "Simulation time:",time()-start_time

subplot(211)
plot(rate.times/ms,rate.smooth_rate(500*ms))
subplot(212)
plot(synapses.W.squeeze(),'.')
show()
