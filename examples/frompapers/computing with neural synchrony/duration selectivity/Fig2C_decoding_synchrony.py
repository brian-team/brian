#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 2C. Decoding synchrony patterns.

Caption (Fig. 2C). Activation of
the postsynaptic assembly as a function of duration (grey: individual
neurons; black: average).

The script Fig2A_synchrony_partition must be run first (it produces a file).
"""
from brian import *
from numpy.random import seed
from params import *
from pylab import cm

Ndur=2000*5 # number of stimulus durations (durations are multiplexed)
sigma=0.1 # More/less noise

best_duration=500*ms

# Read group index for each neuron
f=open('groups'+str(int(best_duration/ms))+'.txt')
group_number=array([int(x) for x in f.read().split(' ')])
nneurons=max(group_number)+1 # one postsynaptic neuron per group
f.close()

# Calculate group size
count=zeros(nneurons) # number of presynaptic neurons for each postsynaptic neuron
for i in range(len(group_number)):
    if group_number[i]!=-1:
        count[group_number[i]]+=1

# Presynaptic neurons
ginh_max=5.
Nx=5 # number of neurons per row
N=Nx*Nx # number of neurons
rest_time=1*second # initial time
eqs='''
dv/dt=(El-v+(gmax*gK+gmax2*gK2+ginh)*(EK-v))/tau : volt
dgK/dt=(gKinf-gK)/tauK : 1 # IKLT
dgK2/dt=-gK2/tauK2 : 1 # Delayed rectifier
gKinf=1./(1+exp((Va-v)/ka)) : 1
ginh = ginh_max*((t>rest_time) & (t<(rest_time+duration))) : 1
tauK : ms
tau : ms
gmax : 1
duration : second
'''

uniform=lambda N:(rand(N)-.5)*2 #uniform between -1 and 1
seed(31418) # Get the same neurons every time
_tauK=400*ms+uniform(N)*tauK_spread
alpha=(El-Vt)/(Vt-EK)
_gmax=alpha*(minx+(maxx-minx)*rand(N))
_tau=30*ms+uniform(N)*tau_spread

neurons=NeuronGroup(N*Ndur,model=eqs,threshold='v>Vt',reset='v=Vr;gK2=1')
neurons.v=Vr
neurons.gK=1./(1+exp((Va-El)/ka))

# Postsynaptic neurons (noisy coincidence detectors)
eqs_post='''
dv/dt=(n-v)/tau_cd : 1
dn/dt=-n/tau_n+sigma*(2/tau_n)**.5*xi : 1
'''
postneurons=NeuronGroup(Ndur*nneurons,model=eqs_post,threshold=1,reset=0)
C=Connection(neurons,postneurons)

# Divide into subgroups, each group corresponds to one postsynaptic neuron with all stimulus durations
postgroup=[]
for i in range(nneurons):
    postgroup.append(postneurons.subgroup(Ndur))

# Connections according to the synchrony partition
group=[]
for i in range(N):
    group.append(neurons.subgroup(Ndur))
    group[i].tauK=_tauK[i]
    group[i].gmax=_gmax[i]
    group[i].tau=_tau[i]
    group[i].duration=linspace(100*ms,1*second,Ndur)
    if group_number[i]>=0:
        C.connect_one_to_one(group[i],postgroup[group_number[i]],weight=1./count[group_number[i]])

spikes=SpikeCounter(postneurons)

run(rest_time+1.1*second,report='text')

# Figure (2C)
window=100*5 # smoothing window
rate=zeros(Ndur-window)
totrate=zeros(Ndur-window)
for i in range(nneurons): # display tuning curve for each neuron, in grey
    count=spikes.count[i*Ndur:(i+1)*Ndur]
    # Smooth
    for j in range(0,len(count)-window):
        rate[j]=mean(count[j:j+window])
    totrate+=rate
    if i<5: # plot only 5 individual curves
        plot((group[0].duration[window/2:-window/2]/ms),rate,'grey',linewidth=1)
# Mean tuning curve
plot((group[0].duration[window/2:-window/2]/ms),totrate/nneurons,'k',linewidth=2)
xlim(100,600)
ylim(0,0.5)
xlabel('Duration (ms)')
ylabel('Spiking probability')
show()
