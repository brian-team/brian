#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 1C,D. Duration selectivity.
(takes about 1 min)

Caption (Fig. 1C,D).
A postsynaptic neuron receives inputs from A and B. It is
more likely to fire when the stimulus in the synchrony receptive field of
A and B.
"""
from brian import *

# Parameters and equations of the rebound neurons
Vt=-55*mV
Vr=-70*mV
El=-35*mV
EK=-90*mV
Va=Vr
ka=5*mV
gmax2=2
tau=20*ms
ginh_max=5.
tauK2=100*ms
N=10000 # number of neurons (= different durations)
rest_time=1*second # initial time
tmin=rest_time-20*ms #for plots
tmax=rest_time+600*ms

eqs='''
dv/dt=(El-v+(gmax*gK+gmax2*gK2+ginh)*(EK-v))/tau : volt
dgK/dt=(gKinf-gK)/tauK : 1 # IKLT
dgK2/dt=-gK2/tauK2 : 1 # Delayed rectifier
gKinf=1./(1+exp((Va-v)/ka)) : 1
duration : second
ginh = ginh_max*((t>rest_time) & (t<(rest_time+duration))) : 1
tauK : ms
gmax : 1
theta : volt # threshold
'''

neurons=NeuronGroup(2*N,model=eqs,threshold='v>theta',reset='v=Vr;gK2=1')
neurons.v=Vr
neurons.theta=Vt
neurons.gK=1./(1+exp((Va-El)/ka))
# Neuron A, duplicated to simulate multiple input durations simultaneously
neuronsA=neurons[:N]
neuronsA.tauK=400*ms
neuronsA.gmax=1
neuronsA.theta=-55*mV
neuronsA.duration=linspace(100*ms,1*second,N)
# Neuron B, duplicated to simulate multiple input durations simultaneously
neuronsB=neurons[N:]
neuronsB.tauK=100*ms
neuronsB.gmax=1.5
neuronsB.theta=-54*mV
neuronsB.duration=linspace(100*ms,1*second,N)

# Noisy coincidence detectors
tau_cd=5*ms
tau_n=tau_cd
sigma=0.2 # noise s.d. in units of the threshold
eqs_post='''
dv/dt=(n-v)/tau_cd : 1
dn/dt=-n/tau_n+sigma*(2/tau_n)**.5*xi : 1
'''
postneurons=NeuronGroup(N,model=eqs_post,threshold=1,reset=0)
CA=IdentityConnection(neuronsA,postneurons,'v',weight=0.5)
CB=IdentityConnection(neuronsB,postneurons,'v',weight=0.5)

spikes=SpikeCounter(postneurons)
M=StateMonitor(postneurons,'v',record=N/3)

run(rest_time+1.1*second,report='text')

# Figure
subplot(121) # Fig. 1C, example trace
plot(M.times/ms,M[N/3],'k')
xlim(1350,1500)
ylim(-.3,1)
xlabel('Time (ms)')
ylabel('V')

subplot(122) # Fig. 1D, duration tuning curve
count=spikes.count
# Smooth the tuning curve
window=200
rate=zeros(len(count)-window)
for i in range(0,len(count)-window):
    rate[i]=mean(count[i:i+window])
plot((neuronsA.duration[window/2:-window/2]/ms)[::10],rate[::10],'k')
xlim(0,1000)
ylim(0,0.5)
xlabel('Duration (ms)')
ylabel('Spiking probability')
show()
