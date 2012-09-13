#!/usr/bin/env python
"""
Wang-Buszaki model
------------------

J Neurosci. 1996 Oct 15;16(20):6402-13.
Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network model.
Wang XJ, Buzsaki G.

Note that implicit integration (exponential Euler) cannot be used, and therefore
simulation is rather slow.
"""
from brian import *

defaultclock.dt=0.01*ms

Cm=1*uF # /cm**2
Iapp=2*uA
gL=0.1*msiemens
EL=-65*mV
ENa=55*mV
EK=-90*mV
gNa=35*msiemens
gK=9*msiemens

eqs='''
dv/dt=(-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL)+Iapp)/Cm : volt
m=alpham/(alpham+betam) : 1
alpham=-0.1/mV*(v+35*mV)/(exp(-0.1/mV*(v+35*mV))-1)/ms : Hz
betam=4*exp(-(v+60*mV)/(18*mV))/ms : Hz
dh/dt=5*(alphah*(1-h)-betah*h) : 1
alphah=0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
betah=1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
dn/dt=5*(alphan*(1-n)-betan*n) : 1
alphan=-0.01/mV*(v+34*mV)/(exp(-0.1/mV*(v+34*mV))-1)/ms : Hz
betan=0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
'''

neuron=NeuronGroup(1,eqs)
neuron.v=-70*mV
neuron.h=1
M=StateMonitor(neuron,'v',record=0)

run(100*ms,report='text')

plot(M.times/ms,M[0]/mV)
show()
