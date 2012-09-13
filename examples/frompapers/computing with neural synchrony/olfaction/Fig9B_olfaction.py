#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 9B.

Caption (Fig. 9B). Top, Fluctuating
concentration of three odors (A: blue, B: red, C: black). Middle, spiking responses of olfactory receptors.
Bottom, Responses of postsynaptic neurons
from the assembly selective to A (blue) and to B (red). Stimuli are presented is sequence: 1) odor A alone,
2) odor B alone, 3) odor B alone with twice
stronger intensity, 4) odor A with distracting odor C (same intensity), 5) odors A and B (same intensity).
'''
from brian import *

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

# Odors
seed(31415) # Get the same neurons every time
intensity=3000.
c1=odor(N)
c2=odor(N)
c0=c1
I1,I2=intensity,intensity

# Odor plumes (fluctuating concentrations)
tau_plume=75*ms
eq_plumes='''
dx/dt=-x/tau_plume+(2./tau_plume)**.5*xi : 1
y=clip(x,0,inf) : 1
'''
plume=NeuronGroup(2,model=eq_plumes) # 2 odors

# Receptor neurons
Fmax=40*Hz # maximum firing rate
tau=20*ms
Imax=1/(1-exp(-1/(Fmax*tau))) # maximum input current

eq_receptors='''
dv/dt=(Imax*hill_function(c)-v)/tau : 1
c : 1  # concentrations (relative to activation constant)
'''

receptors=NeuronGroup(N,model=eq_receptors,threshold=1,reset=0)
receptors.c=c1

@network_operation
def odor_to_nose():
    # Send odor plume to the receptors
    receptors.c=I1*c1*clip(plume.x[0],0,Inf)+I2*c2*clip(plume.x[1],0,Inf)

# Decoder neurons
M=200
taud=8*ms
sigma=.15
eq_decoders='''
dv/dt=-v/taud + sigma*(2/taud)**.5*xi : 1
'''
decoders=NeuronGroup(2*M,model=eq_decoders,threshold=1,reset=0)
# First M neurons encode odor A, next M neurons encode odor B

# Synapses
syn=Connection(receptors,decoders,'v')

# Connectivity according to synchrony partitions
bhalf=.5*(bmin+bmax) # select only those that are well activated
u=2*(log(c1)/log(10)-bhalf)/(bmax-bmin) # normalized binding constants for odor A
for i in range(M):
    which=((u>=i*1./M) & (u<(i+1)*1./M)) # we divide in M groups with similar values
    if sum(which)>0:
        w=1./sum(which) # total synaptic weight for a postsynaptic neuron is 1
    syn[:,i]=w*which

u=2*(log(c2)/log(10)-bhalf)/(bmax-bmin)
for i in range(M): # normalized binding constants for odor B
    which=((u>=i*1./M) & (u<(i+1)*1./M))
    if sum(which)>0:
        w=1./sum(which)
    syn[:,2*M-1-i]=w*which

# Record odor concentration and output spikes
O=StateMonitor(plume,'y',record=True)
S=SpikeMonitor(receptors)
S2=SpikeMonitor(decoders)

print "Odor A"
I1,I2=intensity,0
run(2*second)
print "Odor B"
I1,I2=0,intensity
run(2*second)
print "Odor B x2"
I1,I2=0,2*intensity
run(2*second)
print "Odor A + odor C"
I1,I2=intensity,intensity
old_c2=c2
c2=odor(N) # different odor
run(2*second)
print "Odor A + odor B"
I1,I2=intensity,intensity
c2=old_c2
run(2*second)

t=O.times/ms

# Figure (9B)
subplot(311) # odor fluctuations
plot(t[t<2000],O[0][t<2000],'b')
plot(t[(t>=2000) & (t<4000)],O[1][(t>=2000) & (t<4000)],'r')
plot(t[(t>=4000) & (t<6000)],2*O[1][(t>=4000) & (t<6000)],'r')
plot(t[(t>=6000) & (t<8000)],O[0][(t>=6000) & (t<8000)],'b')
plot(t[(t>=6000) & (t<8000)],O[1][(t>=6000) & (t<8000)],'k')
plot(t[(t>=8000) & (t<10000)],O[1][(t>=8000) & (t<10000)],'r')
plot(t[(t>=8000) & (t<10000)],O[0][(t>=8000) & (t<10000)],'b')
xlim(0,10000)
xticks([])
subplot(312)
raster_plot(S)
xlim(0,10000)
ylim(2500,2600) # 100 random neurons
xticks([])
subplot(313)
raster_plot(S2)
ylim(100,300)
xlim(0,10000)

show()
