from brian import *

# Simulation control
Npulses=1000
Ntest=20
record_period=1*second
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
sigma=0.2 # noise s.d.
tau_cd=5*ms # could be too slow?
tau_n=tau_cd # slow noise (I need to calculate sigma_v vs. sigma_n)
refractory=0*ms
#Ee=5 # relative to threshold-rest
# Connections
Nsynapses=10 # synapses per neuron (was 10)
#w0=.2 # initial synaptic strength
w0=lambda i,j:.6*rand()*.2
# STDP
a_pre=0.02
b_post=-a_pre
b_pre=-b_post*.1
tau_pre=tau_cd
delay_stdp=0.2*ms
