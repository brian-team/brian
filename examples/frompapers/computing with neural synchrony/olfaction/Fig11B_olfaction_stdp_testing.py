#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 11B. Learning to recognize odors.

Caption (Fig. 11B). After learning, responses
of postsynaptic neurons, ordered by tuning ratio, to odor A (blue) and odor B (red),
with an increasing concentration (0.1 to 10, where 1 is odor
concentration in the learning phase).

Run the other file first: Fig11B_olfaction_stdp_learning.py
'''
from brian import *
from params import *
import numpy
from scipy.sparse import lil_matrix

bmin,bmax=-7,-1

# Loads information from the STDP simulation
t,odor=numpy.load("stimuli.npy").T
W=numpy.load("weights.npy")
spikes_out=numpy.load("spikesout.npy")
weights=W[-1,:,:] # final weights

# Analyze selectivity at the end of the STDP simulation
ispikes=spikes_out[:,0] # indexes of neurons that spiked
tspikes=spikes_out[:,1] # spike timings
# Select only the end of the STDP simulation
end=tspikes>.8*max(tspikes)
ispikes=ispikes[end]
tspikes=tspikes[end]

odors=odor[digitize(tspikes,t)-1] # odor (0/1) presented at the time of spikes

tuning=zeros(30) # Tuning ratio of the postsynaptic neurons
n0,n1=zeros(30),zeros(30) # number of spikes for odor 0 and for odor 1
for k in range(len(tuning)):
    o=odors[ispikes==k]
    n0[k]=sum(o==0)
    n1[k]=sum(o==1)
    tuning[k]=n0[k]*1./(n0[k]+n1[k])

# Sort the postsynaptic neurons by odor tuning
weights=weights[:,argsort(tuning)]

'''
Run the simulation
'''
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
plume=NeuronGroup(2,model=eq_plumes) # 1 odor

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
    receptors.c=I1*c1*clip(plume.x[0],0,Inf)+I2*c2*clip(plume.x[0],0,Inf)

odors=[odor(N),odor(N)]
c1,c2=odors

# Decoder neurons
M=len(tuning)
eq_decoders='''
dv/dt=-v/taud + sigma*(2/taud)**.5*xi : 1
'''
decoders=NeuronGroup(M,model=eq_decoders,threshold=1,reset=0)
S2=SpikeMonitor(decoders)

# Synapses
syn=Connection(receptors,decoders,'v')
for i in range(len(decoders)):
    for j in weights[:,i].nonzero()[0]:
        syn[j,i]=weights[j,i]

# Run
I1,I2=intensity,0
print "Started"
# Odor A, increasing concentration
for I1 in intensity*exp(linspace(log(.1),log(10),20)):
    run(1*second,report="text")
I1=0
# Odor B, increasing concentration
for I2 in intensity*exp(linspace(log(.1),log(10),20)):
    run(1*second,report="text")

# Figure (11B)
spikes=array(S2.spikes) # i,t
n,t=spikes[:,0],spikes[:,1]
subplot(211) # Raster plot
plot(t,n,'k.')
subplot(212) # Odor concentrations
semilogy(linspace(0,20,20),exp(linspace(log(.1),log(10),20)),'b')
semilogy(linspace(20,40,20),exp(linspace(log(.1),log(10),20)),'r')
plot([0,40],[1,1],'k--')
show()
