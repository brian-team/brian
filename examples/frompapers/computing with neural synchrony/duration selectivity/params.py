#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Duration selectivity, parameters
'''
from brian import *

# Simulation control
Npulses=5000
Ntest=20
record_period=15*second
rest_time=200*ms
# Encoding neurons
Vt=-55*mV
Vr=-70*mV
El=-35*mV
EK=-90*mV
Va=Vr
ka=5*mV
gmax2=2
tauK2=300*ms
N=100 # number of encoding neurons
Nout=30 # number of decoding neurons
ginh_max=5.
tauK_spread=200*ms
tau_spread=20*ms
minx=1.7 # range of gmax for K+
maxx=2.5
# Coincidence detectors
sigma=0.1 # noise s.d.
tau_cd=5*ms
tau_n=tau_cd # slow noise
refractory=0*ms
# Connections
Nsynapses=5 # synapses per neuron
w0=lambda i,j:rand()
# STDP
factor=0.05
a_pre=.06*factor
b_post=-1.*factor
b_pre=0.1*factor
tau_pre=tau_cd
