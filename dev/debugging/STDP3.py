'''
Network of excitatory IF neurons with additive STDP
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
gmax=0.15
dA_pre=gmax*.005
dA_post=-dA_pre*1.06
N=100
w0=gmax
sigmae=gmax

eqs_neurons='''
dv/dt=(ge*(Ee-v)+El-v)/taum : volt
dge/dt=-ge/taue+.5*sigmae*xi/taue**.5 : 1
dA_post/dt=-A_post/tau_post : 1
dA_pre/dt=-A_pre/tau_pre : 1
'''

neurons=NeuronGroup(N,model=eqs_neurons,threshold=vt,reset=vr)
synapses=Connection(neurons,neurons,'ge',structure='dense')
synapses.connect_full(neurons,neurons,weight=w0)
neurons.v=rand(N)*(vt-vr)+vr
neurons.ge=rand(N)*w0*10

S=SpikeMonitor(neurons)

run(100*ms)
raster_plot(S)
show()
