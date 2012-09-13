#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 2A. Synchrony partition for duration selective neurons.

Caption (Fig. 2A)
Decoding synchrony patterns in a heterogeneous
population.
Color represents the latency of the spike produced by each neuron
responding to the stimulus (white if the neuron did not spike). Thus,
neurons with the same color are synchronous for that specific stimulus
(duration). The population can be divided in groups of synchronous
neurons (i.e., with the same color), forming the "synchrony partition".
Circled neurons belong to the same synchronous group.

This script calculates and displays the synchrony partition for one particular duration.
It also saves the results in file, that is required by the script Fig2C_decoding_synchrony.
The synchrony partition is calculated empirically, by simulating the responses of the neurons at
the specific inhibitory duration
and grouping neurons that respond in synchrony (+- 2 ms).
"""
from brian import *
from numpy.random import seed
from params import *
from pylab import cm

# Graphics
radius=.15
selected_neuron=7
# Parameters
ginh_max=5.
Nx=5                # number of neurons per row
N=Nx*Nx             # number of neurons
rest_time=1*second  # initial time
duration=500*ms
delta_t=2*ms        # Size of synchronous groups (maximum time difference)

# Duration-selective neurons
eqs='''
dv/dt=(El-v+(gmax*gK+gmax2*gK2+ginh)*(EK-v))/tau : volt
dgK/dt=(gKinf-gK)/tauK : 1 # IKLT
dgK2/dt=-gK2/tauK2 : 1 # Delayed rectifier
gKinf=1./(1+exp((Va-v)/ka)) : 1
ginh = ginh_max*((t>rest_time) & (t<(rest_time+duration))) : 1
tauK : ms
tau : ms
gmax : 1
'''

uniform=lambda N:(rand(N)-.5)*2 # uniform between -1 and 1
seed(31418) # Get the same neurons every time

neurons=NeuronGroup(N,model=eqs,threshold='v>Vt',reset='v=Vr;gK2=1')
neurons.v=Vr
neurons.gK=1./(1+exp((Va-El)/ka))
neurons.tauK=400*ms+uniform(N)*tauK_spread
alpha=(El-Vt)/(Vt-EK)
neurons.gmax=alpha*(minx+(maxx-minx)*rand(N))
neurons.tau=30*ms+uniform(N)*tau_spread

spikes=SpikeMonitor(neurons)

run(rest_time+1.1*second)

# Calculate first spike time of each neuron
times=zeros(N) # First spike time of each neuron
times[:]=Inf # Inf means: no response, or response before the start of the stimulus
blacklist=[] # neurons that fire spontaneously
for i,t in spikes.spikes:
    if times[i]==Inf:
        times[i]=t-duration-rest_time
        if times[i]<0:
            blacklist.append(i)
times[blacklist]=Inf
tmin,tmax=min(times[times!=Inf]),max(times[times!=Inf])
# Color of each neuron between 0 and 1
color=(times-tmin)/(tmax+1e-10-tmin) # (to avoid zero division)

# Assign groups; each responding neuron gets a group number
group_size=delta_t/(tmax-tmin) # size of a group, as a proportion of the timing range
group_number=array(color/group_size,dtype=int)
group_number[color==Inf]=-1

# Get the size of each group
count=zeros(max(group_number)+1) # number of neurons in each group
for i in range(len(group_number)):
    if group_number[i]!=-1:
        count[group_number[i]]+=1

selected_group=group_number[selected_neuron]

# Display the synchrony partition (Fig. 2A)
axes(frameon=False)
axis('scaled')
xticks([])
yticks([])
i=0
for y in linspace(0,1,Nx):
    for x in linspace(0,1,Nx):
        if color[i]!=Inf:
            if group_number[i]==selected_group:
                w=4
                ec="k" # edge color
            else:
                w=1
                ec='k'
            cir=Circle((x,y),radius,fc=cm.jet(color[i]),linewidth=w,ec=ec)
        else:
            cir=Circle((x,y),radius,fc='w')
        i+=1
        gca().add_patch(cir)
xlim(0-2*radius,1+2*radius)
ylim(0-2*radius,1+2*radius)

# Remove groups with fewer than two neurons and recalculate group numbers
for i in range(len(group_number)):
    if group_number[i]>=0:
        if count[group_number[i]]>=2:
            group_number[i]=sum(count[:group_number[i]]>=2)
        else:
            group_number[i]=-1
            
# Save assignment to groups
f=open('groups'+str(int(duration/ms))+'.txt','w')
f.write(' '.join([str(x) for x in group_number]))
f.close()

show()
