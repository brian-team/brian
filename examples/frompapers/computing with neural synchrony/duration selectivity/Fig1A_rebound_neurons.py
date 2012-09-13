#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 1A, B.

Caption (Fig. 1A,B) A, When neuron A is
hyperpolarized by an inhibitory input (top), its low-voltage-activated
K channels slowly close (bottom), which makes the neuron fire when
inhibition is released (neuron models are used in this and other figures).
B, Spike latency is negatively correlated with the duration of inhibition
(black line).
"""
from brian import *

# Parameters and equations of the rebound neurons
Vt=-55*mV
Vr=-70*mV
El=-35*mV
EK=-90*mV
Va=Vr
ka=5*mV
gmax=1
gmax2=2
tau=20*ms
ginh_max=5.
tauK=400*ms
tauK2=100*ms
N=100 # number of neurons (= different durations, for plot 1B)
plotted_neuron=N/4
rest_time=1*second # initial time (to start at equilibrium)
tmin=rest_time-20*ms # for plots
tmax=rest_time+600*ms

eqs='''
dv/dt=(El-v+(gmax*gK+gmax2*gK2+ginh)*(EK-v))/tau : volt
dgK/dt=(gKinf-gK)/tauK : 1 # IKLT
dgK2/dt=-gK2/tauK2 : 1 # Delayed rectifier
gKinf=1./(1+exp((Va-v)/ka)) : 1
duration : second # duration of inhibition, varies across neurons
ginh = ginh_max*((t>rest_time) & (t<(rest_time+duration))) : 1
'''

neurons=NeuronGroup(N,model=eqs,threshold='v>Vt',reset='v=Vr;gK2=1')
neurons.v=Vr
neurons.gK=1./(1+exp((Va-El)/ka))
neurons.duration=linspace(100*ms,1*second,N)
M=StateMonitor(neurons,'v',record=plotted_neuron)
Mg=StateMonitor(neurons,'gK',record=plotted_neuron)
spikes=SpikeMonitor(neurons)

run(rest_time+1.1*second)

M.insert_spikes(spikes) # draw spikes for a nicer display

# Figure
subplot(221) # Fig. 1A, top
plot((M.times-tmin)/ms,M[plotted_neuron]/mV,'k')
xlim(0,(tmax-tmin)/ms)
ylabel('V (mV)')
subplot(223) # Fig. 1A, bottom
plot((Mg.times-tmin)/ms,Mg[plotted_neuron],'k')
xlim(0,(tmax-tmin)/ms)
xlabel('Time (ms)')
ylabel('g/gmax')

subplot(122) # Fig. 1B
times=array([t-neurons.duration[i]*second-rest_time for i,t in spikes.spikes])
duration=array([neurons.duration[i]*second for i,_ in spikes.spikes])
plot(duration/ms,times/ms,'k')
xlabel('Duration (ms)')
ylabel('Latency (ms)')
show()
