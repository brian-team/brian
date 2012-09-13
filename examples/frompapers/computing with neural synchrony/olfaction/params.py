#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Parameters for the olfactory model with STDP
'''
from brian import *

N=5000
# Coincidence detectors
sigma=.15
taud=8*ms
# Connections
Nsynapses=50
w0=150./(0.02*N)
# STDP
factor=0.05
a_pre=0.06*factor
b_post=-1.*factor
tau_pre=3*ms
# Intrinsic plasticity: non-specific weight increase
IP_period=10*ms
IP_rate=-b_post*5*Hz # target firing rate = 5 Hz
# Simulation control
record_period=1*second
duration=100*second
