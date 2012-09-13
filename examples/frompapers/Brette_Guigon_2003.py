#!/usr/bin/env python
"""
Reliability of spike timing
---------------------------
Adapted from Fig. 10D,E of
Brette R and E Guigon (2003). Reliability of Spike Timing Is a General Property
of Spiking Model Neurons. Neural Computation 15, 279-308.

This shows that reliability of spike timing is a generic property of spiking
neurons, even those that are not leaky.
This is a non-physiological model which can be leaky or anti-leaky depending
on the sign of the input I.

All neurons receive the same fluctuating input, scaled by a parameter p that
varies across neurons. This shows:

1. reproducibility of spike timing
2. robustness with respect to deterministic changes (parameter)
3. increased reproducibility in the fluctuation-driven regime (input crosses
   the threshold)
"""
from brian import *

N=500
tau=33*ms
taux=20*ms
sigma=0.02

eqs_input='''
B=2./(1+exp(-2*x))-1 : 1
dx/dt=-x/taux+(2/taux)**.5*xi : 1
'''

eqs='''
dv/dt=(v*I+1)/tau + sigma*(2/tau)**.5*xi : 1
I=0.5+3*p*B : 1
B : 1
p : 1
'''

input=NeuronGroup(1,eqs_input)
neurons=NeuronGroup(N,eqs,threshold=1,reset=0)
neurons.p=linspace(0,1,N)
neurons.v=rand(N)
neurons.B=linked_var(input,'B')

M=StateMonitor(input,'B',record=0)
S=SpikeMonitor(neurons)

run(1000*ms)

subplot(211) # The input
plot(M.times/ms,M[0])
subplot(212)
raster_plot(S)
plot([0,1000],[250,250],'r')
show()
