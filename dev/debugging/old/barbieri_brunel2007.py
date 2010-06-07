'''
Adapted from
Irregular persistent activity induced by synaptic excitatory feedback
F Barbieri, N Brunel - Frontiers in Computational Neuroscience, 2007

R Brette 2009
'''
from brian import *

theta=20*mV
Vr=15*mV
tau=5*ms
tau_rp=2*ms
N=800
nu_sp=3*Hz
mu_ext=10*mV # (?)
sigma_ext=8*mV
tau_r=.05*ms
tau_rn=2*ms
tau_d=5*ms
tau_dn=100*ms
gamma=.9
u=0.5
tau_rec=160*ms
lambd=0.1
D=1*ms # delay (?)
J=35*mV # (?)

#defaultclock.dt=.001*ms

eqs='''
dv/dt=(-v+I_rec+mu_ext*tau**.5*xi)/tau : volt
I_rec=s+z : volt
ds/dt=(x-s)/tau_d : volt
dx/dt=-x/tau_r : volt
dz/dt=(h-z)/tau_dn : volt
dh/dt=-h/tau_rn : volt
'''

neurons=NeuronGroup(N, model=eqs, threshold=theta, reset=Vr, refractory=tau_rp)
synapses_ampa=Connection(neurons, neurons, 'x', weight=(1-gamma)*tau/(tau_r*N)*J)
synapses_nmda=Connection(neurons, neurons, 'h', weight=gamma*tau/(tau_rn*N)*J)
stp1=STP(synapses_ampa, taud=tau_rec, tauf=.1*ms, U=u)
stp2=STP(synapses_nmda, taud=tau_rec, tauf=.1*ms, U=u)

spikes=SpikeMonitor(neurons)

run(1*second)
#run(500*ms)
raster_plot(spikes)
print spikes.nspikes*1./N
show()
