#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 6. Shared variability.

This example shows that synchrony may be reproducible without individual responses being reproducible,
because of shared variability (here due to a common input).

Caption (Fig 6).
Neurons A and B receive the same stimulus-driven input, neuron C
receives a different one. The stimuli are identical in all trials but all
neurons receive a shared input that varies between trials. Each neuron
also has a private source of noise.
Top, Responses of neurons A (black), B
(red) and C (blue) in 25 trials, with a signal-to-noise ratio (SNR) of 10 dB
(shared vs. private).
Bottom left, The shuffled autocorrelogram of neuron A
indicates that spike trains are not reproducible at a fine timescale.
Botto right, Nevertheless, the average cross-correlogram between A and B shows
synchrony at a millisecond timescale, which does not appear between A
and C.
"""
from brian import *

# Inputs
N=100 # number of trials
tau=10*ms
sigma=0.7

eqs_input='''
dx/dt=-x/tau+(2/tau)**.5*xi : 1
'''

input=NeuronGroup(N+2,eqs_input)
shared=input[:N]        # different in all trials, but common to all neurons
stimulus1=input[N:N+1]  # identical in all trials
stimulus2=input[N+1:]   # identical in all trials

# Neurons
taum=10*ms
#sigma_noise=.05
duration=3000*ms

#sigma=sigma*sqrt(2.)
SNRdB=10.
SNR = 10.**(SNRdB/10.)
Z=sigma*sqrt((taum+tau)/(tau*(SNR**2+1))) # normalizing factor
#print Z,Z*SNR
#Z=sigma*sqrt(1./(SNR**2+1))
eqs_neurons='''
dv/dt=(Z*(SNR*I+n)-v)/taum: 1
dn/dt=-n/tau+(2./tau)**.5*xi : 1
I : 1
'''

neuron=NeuronGroup(3*N,eqs_neurons,threshold=1,reset=0)
neuronA=neuron[:N]
neuronB=neuron[N:2*N]
neuronC=neuron[2*N:]
neuron.n=randn(len(neuron))

@network_operation
def inject():
    neuronA.I=shared.x+stimulus1.x
    neuronB.I=shared.x+stimulus1.x
    neuronC.I=shared.x+stimulus2.x

spikes=SpikeMonitor(neuron)

run(duration,report='text')

# Figure
figure()
# Raster plot
subplot(211) # Fig. 6B
i,t=zip(*[(i,t) for i,t in spikes.spikes if (i<25)])
plot(t,array(i)+50,'.k')
i,t=zip(*[(i,t) for i,t in spikes.spikes if (i>=N) & (i<N+25)])
plot(t,array(i)-N+25,'.r')
i,t=zip(*[(i,t) for i,t in spikes.spikes if (i>=2*N) & (i<2*N+25)])
plot(t,array(i)-2*N,'.b')
ylim(0,75)
xlabel('Time (s)')
ylabel('Trials')

# Cross-correlograms (CC)
width=100*ms
bin=1*ms
spikes=spikes.spiketimes
C_AB=correlogram(spikes[0],spikes[N],width=width,T=duration)
for i in range(1,N):
    C_AB+=correlogram(spikes[i],spikes[N+i],width=width,T=duration)
C_AC=correlogram(spikes[0],spikes[2*N],width=width,T=duration)
for i in range(1,N):
    C_AC+=correlogram(spikes[i],spikes[2*N+i],width=width,T=duration)

# Shuffled auto-correlogram (SAC)
C=0*C_AB
for i in range(0,N):
    for j in range(0,N):
        if i!=j:
            C+=correlogram(spikes[i],spikes[j],width=width,T=duration)

lag=(arange(len(C))-len(C)/2)*bin

subplot(223) # Fig. 6C
plot(lag/ms,C/(bin*N*(N-1)),'k')
ylim(0,1.1*max(C_AB/(bin*N)))
xlabel('Lag (ms)')
ylabel('Coincidences')

subplot(224) # Fig. 6D
plot(lag/ms,C_AB/(bin*N),'k') # A vs. B
plot(lag/ms,C_AC/(bin*N),'r') # A vs. C
ylim(0,1.1*max(C_AB/(bin*N)))
xlabel('Lag (ms)')
ylabel('Coincidences')

show()
