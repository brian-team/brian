'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)
'''
from brian import *
from time import time

taum=20*ms
tau_pre=20*ms
tau_post=tau_pre*5
Ee=0*mV
vt=-54*mV
vr=-60*mV
El=-74*mV
taue=5*ms
taui=5*ms
gmax=0.015
gin=0.05
Fin=200*10*Hz
dA_pre=gmax*.005*3
dA_post=-dA_pre*tau_pre/tau_post*1.05

eqs_neurons='''
dv/dt=(ge*(Ee-v)+El-v)/taum : volt
dge/dt=-ge/taue : 1
'''

#eqs_stdp='''
#dA_pre/dt=-A_pre/tau_pre : 1
#dA_post/dt=-A_post/tau_post : 1
#'''

input=PoissonGroup(1000,rates=15*Hz)
neurons=NeuronGroup(1,model=eqs_neurons,threshold=vt,reset=vr)
synapses=Connection(input,neurons,'ge',weight=rand(len(input),len(neurons))*gmax,
                    structure='dense')
neurons.v=vr

#stdp=STDP(synapses,eqs=eqs_stdp,pre='A_pre+=dA_pre;w+=A_post',
#          post='A_post+=dA_post;w+=A_pre',bounds=(0,gmax))
stdp=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)

rate=PopulationRateMonitor(neurons)

start_time=time()
run(100*second)
print "Simulation time:",time()-start_time

subplot(311)
plot(rate.times/ms,rate.smooth_rate(100*ms))
subplot(312)
plot(synapses.W.todense(),'.')
subplot(313)
hist(synapses.W.todense(),20)
show()
