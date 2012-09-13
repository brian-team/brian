#!/usr/bin/env python
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 4D. STDP in the duration model.
(very long simulation, default: 5000 stimuli)

Caption (Fig. 4D). Temporal evolution of the synaptic weights of the neuron corresponding to the blue curves in Fig. 4C.

The script runs the simulation with STDP for a long time, then displays the evolution of synaptic weights for one neuron.
"""
from brian import *
from pylab import cm
from numpy.random import seed
from brian.experimental.connectionmonitor import *
import numpy
from params import *

# Rebound neurons
eqs='''
dv/dt=(El-v+(gmax*gK+gmax2*gK2+ginh)*(EK-v))/tau : volt
dgK/dt=(gKinf-gK)/tauK : 1 # IKLT
dgK2/dt=-gK2/tauK2 : 1 # Delayed rectifier
gKinf=1./(1+exp((Va-v)/ka)) : 1
tauK : ms
tau : ms
gmax : 1
ginh : 1
'''

uniform=lambda N:(rand(N)-.5)*2 #uniform between -1 and 1
seed(31415) # Get the same neurons every time

neurons=NeuronGroup(N,model=eqs,threshold='v>Vt',reset='v=Vr;gK2=1')
neurons.v=Vr
neurons.gK=1./(1+exp((Va-El)/ka))
neurons.tauK=400*ms+uniform(N)*tauK_spread
alpha=(El-Vt)/(Vt-EK)
neurons.gmax=alpha*(minx+(maxx-minx)*rand(N))
neurons.tau=30*ms+uniform(N)*tau_spread

# Store the value of state variables at rest
print "Calculate resting state"
run(2*second)
rest=zeros(neurons._S.shape)
rest[:]=neurons._S

# Postsynaptic neurons (noisy coincidence detectors)
eqs_post='''
dv/dt=(n-v)/tau_cd : 1
dn/dt=-n/tau_n+sigma*(2/tau_n)**.5*xi : 1
'''
postneurons=NeuronGroup(Nout,model=eqs_post,threshold=1,reset=0,refractory=refractory)

# Random connections between pre and post-synaptic neurons
C=Connection(neurons,postneurons,'v',sparseness=Nsynapses*1./N,weight=w0)

# STDP
eqs_stdp='''
dApre/dt=-Apre/tau_pre : 1
Apost : 1
'''
pre='''
Apre+=a_pre
w+=0 #b_pre*w
'''
post='''
Apost+=0
w+=Apre+b_post*w
'''
stdp=STDP(C,eqs_stdp,pre=pre,post=post,wmax=Inf)

# Record the evolution of synaptic weights
MC=ConnectionMonitor(C,store=True,clock=EventClock(dt=record_period))

print "Learning..."
# Series of inhibitory pulses
for i in range(Npulses):
    print "pulse",i+1
    duration=200*ms+rand()*300*ms # random stimulus duration
    neurons.ginh=ginh_max
    run(duration)
    C.W.alldata[:]=C.W.alldata+C.W.alldata*b_pre # homeostasis (synaptic scaling)
    neurons.ginh=0
    run(rest_time) # let neurons spike
    neurons._S[:]=rest # reset (to save time)

# Figure (4D)
neuron=0
wsave=[(t,M.todense()) for (t,M) in MC.values]
W=array(zip(*wsave)[1])
weights=W[:,:,neuron]

# Evolution of all synaptic weights for this neuron
for i in range(weights.shape[1]):
    plot(arange(len(weights[:,i,]))*record_period,weights[:,i,],'k')
xlim(0,weights.shape[0]*float(record_period))
ylim(0,1)

show()
