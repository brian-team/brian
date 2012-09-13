#!/usr/bin/env python
'''
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 5D, left.

Caption (Fig 5D). Responses of a noisy integrate-and-fire model in repeated trials.

Protocol: neuron receives input = signal + noise, both O-U processes, signal
is identical in all trials (frozen noise). The total variance is held fixed.
Signal-to-noise ratio is 3 in this simulation.
'''
from brian import *

# The common noisy input
tau_noise=5*ms
input=NeuronGroup(1,model='dx/dt=-x/tau_noise+(2./tau_noise)**.5*xi:1')

# The noisy neurons receiving the same input + independent noise
tau=10*ms
SNR=3. # signal to noise ratio
sigma=.5 # total input amplitude
Z=sigma*sqrt((tau_noise+tau)/(tau_noise*(SNR**2+1))) # normalizing factor
eqs_neurons='''
dx/dt=(Z*(SNR*I+u)-x)/tau:1
du/dt=-u/tau_noise+(2./tau_noise)**.5*xi:1
I : 1
'''
neurons=NeuronGroup(25,model=eqs_neurons,threshold=1,reset=0,refractory=5*ms)
neurons.x=rand(25) # random initial conditions
neurons.I=linked_var(input,'x')
spikes=SpikeMonitor(neurons)

run(2*second)

# Figure
raster_plot(spikes)
show()
