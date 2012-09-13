#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 11B. Learning to recognize odors.
(long simulation)

Caption (Fig. 11B). After learning, responses
of postsynaptic neurons, ordered by tuning ratio, to odor A (blue) and odor B (red),
with an increasing concentration (0.1 to 10, where 1 is odor
concentration in the learning phase).

After this script, run the other file: Fig11B_olfaction_stdp_testing.py.
'''
from brian import *
from params import *
from brian.experimental.connectionmonitor import *
import numpy

bmin,bmax=-7,-1

def odor(N):
    # Returns a random vector of binding constants
    return 10**(rand(N)*(bmax-bmin)+bmin)

def hill_function(c,K=1.,n=3.):
    '''
    Hill function:
    * c = concentration
    * K = half activation constant (choose K=1 for relative concentrations)
    * n = Hill coefficient
    '''
    return (c**n)/(c**n+K**n)

N=5000 # number of receptors

seed(31415) # Get the same neurons every time
intensity=3000.

# Odor plumes
tau_plume=75*ms
eq_plumes='''
dx/dt=-x/tau_plume+(2./tau_plume)**.5*xi : 1
y=clip(x,0,inf) : 1
'''
plume=NeuronGroup(1,model=eq_plumes) # 1 odor

# Receptor neurons
Fmax=40*Hz # maximum firing rate
tau=20*ms
Imax=1/(1-exp(-1/(Fmax*tau))) # maximum input current

eq_receptors='''
dv/dt=(Imax*hill_function(c)-v)/tau : 1
c : 1  # concentrations (relative to activation constant)
'''

receptors=NeuronGroup(N,model=eq_receptors,threshold=1,reset=0)

@network_operation
def odor_to_nose():
    # Send odor plume to the receptors
    receptors.c=I1*c1*clip(plume.x[0],0,Inf)

odors=[odor(N),odor(N)] # two odors
c1=odors[0]
stimuli=[]
# A random odor is presented every 200 ms
@network_operation(clock=EventClock(dt=200*ms))
def change_odor():
    global c1
    nodor=randint(len(odors))
    c1=odors[nodor]
    stimuli.append((float(defaultclock.t),float(nodor)))

# Decoder neurons
M=30
eq_decoders='''
dv/dt=-v/taud + sigma*(2/taud)**.5*xi : 1
'''
decoders=NeuronGroup(M,model=eq_decoders,threshold=1,reset=0)
S2=SpikeMonitor(decoders)

# Random synapses
syn=Connection(receptors,decoders,'v',sparseness=Nsynapses*1./N,weight=w0)

# STDP
eqs_stdp='''
dApre/dt=-Apre/tau_pre : 1
Apost : 1
'''
pre='''
Apre+=a_pre
#w+=0
'''
post='''
Apost+=0
w+=Apre+b_post*w
'''
stdp=STDP(syn,eqs_stdp,pre=pre,post=post,wmax=Inf)
MC=ConnectionMonitor(syn,store=True,clock=EventClock(dt=record_period))

@network_operation(EventClock(dt=IP_period))
def intrinsic_plasticity(): # synaptic scaling
    # Increases weights of all synapses
    syn.W.alldata+=syn.W.alldata*IP_rate*IP_period

# Record the evolution of weights
weights=[]
@network_operation(EventClock(dt=record_period))
def recordW():
    Z=syn.W[:,0].copy()
    weights.append(Z)

I1=intensity
print "Started"
run(duration,report="text")

# Save data
wsave=[(t,M.todense()) for (t,M) in MC.values]
numpy.save("weights.npy",array(zip(*wsave)[1])) # 3D array (t,i,j)
numpy.save("spikesout.npy",array(S2.spikes))
numpy.save("stimuli.npy",array(stimuli))
