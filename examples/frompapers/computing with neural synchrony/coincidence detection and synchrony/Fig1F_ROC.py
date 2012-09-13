#!/usr/bin/env python
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 1F.
(simulation takes about 10 mins)

Coincidence detection is seen as a signal detection problem, that of detecting a given depolarization in a background
of noise, within one characteristic time constant. The choice of the spike threshold implements a particular trade-off
between false alarms (depolarization was due to noise) and misses (depolarization was not detected).

Caption (Fig 1F). Receiver-operation characteristic (ROC) for one level of noise,
obtained by varying the threshold (black curve). The hit rate is the
probability that the neuron fires within one integration time constant t
when depolarized by Dv, and the false alarm rate is the firing
probability without depolarization. The corresponding theoretical
curve, with sensitivity index d' =Dv/sigma, is shown in red.
"""
from brian import *
from scipy.special import erf

def spike_probability(x): # firing probability for unit variance and zero mean, and threshold = x
    return .5*(1-erf(x/sqrt(2.)))

tau_cd=5*ms     # membrane time constant (cd for coincidence detector)
tau_n=tau_cd    # input is an Ornstein-Uhlenbeck process with the same time constant as the membrane time constant
T=3*tau_n       # neurons are depolarized by w at regular intervales, T is the spacing
Nspikes=10000   # number of input spikes
T0=T*Nspikes    # initial period without inputs, to calculate the false alarm rate
N=500           # number of neurons, each neuron has a different threshold between 0. and 3.
w=1             # synaptic weight (depolarization)
sigma=1.        # input noise s.d.
sigmav=sigma*sqrt(tau_n/(tau_n+tau_cd)) # noise s.d. on the membrane potential
print "d'=",1./sigmav # discriminability index

# Integrate-and-fire neurons
eqs='''
dv/dt=(sigma*n-v)/tau_cd : 1
dn/dt=-n/tau_n+(2/tau_n)**.5*xi : 1
vt : 1 # spike threshold
'''
neurons=NeuronGroup(N,model=eqs,threshold='v>vt',reset='v=0',refractory=tau_cd)
neurons.vt=linspace(0.,3,N) # spike threshold varies across neurons
counter=SpikeCounter(neurons)

# Inputs are regular spikes, starting at T0
input=SpikeGeneratorGroup(1,[(0,n*T+T0) for n in range(Nspikes)])
C=Connection(input,neurons,'v',weight=w)

# Calculate the false alarm rate
run(T0,report='text')
FR=tau_cd*counter.count*1./T0
# Calculate the hit rate
counter.reinit()
run(Nspikes*T,report='text')
HR=counter.count*1./Nspikes-FR*(T-tau_cd)/tau_cd

# Prediction based on Gaussian statistics
FRpred=spike_probability(neurons.vt/sigmav)
HRpred=spike_probability((neurons.vt-w)/sigmav)

# Figure
plot(FR*100,HR*100,'k')          # simulations
plot(FRpred*100,HRpred*100,'r')  # theoretical predictions
plot([0,100],[0,100],'k--')
plot([0,100],[50,50],'k--')
xlim(0,100)
ylim(0,100)
xlabel('False alarm rate (%)')
ylabel('Hit rate (%)')

show()
