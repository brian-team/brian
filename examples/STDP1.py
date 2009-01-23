'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000)
'''
from brian import *
from time import time

taum=20*ms
tau_post=20*ms
tau_pre=20*ms
Ee=0*mV
vt=-54*mV
vr=-60*mV
El=-70*mV
taue=5*ms
gmax=0.015
dA_pre=gmax*.005
dA_post=-dA_pre*1.05

eqs_neurons='''
dv/dt=(ge*(Ee-v)+El-v)/taum : volt
dge/dt=-ge/taue : 1
'''

eqs_stdp='''
dA_pre/dt=-A_pre/tau_pre : 1
dA_post/dt=-A_post/tau_post : 1
'''

input=PoissonGroup(1000,rates=10*Hz)
neurons=NeuronGroup(1,model=eqs_neurons,threshold=vt,reset=vr)
synapses=Connection(input,neurons,'ge')
synapses.connect(input,neurons,rand(len(input),len(neurons))*gmax)
neurons.v=vr

#stdp=STDP(synapses,eqs=eqs_stdp,pre='A_pre+=dA_pre;w+=A_post',
#          post='A_post+=dA_post;w+=A_pre',bounds=(0,gmax))
stdp=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)

rate=PopulationRateMonitor(neurons)

start_time=time()
run(2*second)
print "Simulation time:",time()-start_time

subplot(211)
plot(rate.times/ms,rate.smooth_rate(100*ms))
subplot(212)
plot(synapses.W.todense(),'.')
show()
