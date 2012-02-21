"""
Naundorf et al. (2006), Nature
Unique features...

Cooperative model of spike initiation.

Note that implicit integration is not possible here.
"""
from brian import *

defaultclock.dt=0.005*ms

Cm=1*uF # /cm**2
Iapp=2*uA
gL=0.1*msiemens
EL=-65*mV
ENa=55*mV
EK=-90*mV
gNa=35*msiemens
gK=9*msiemens
KJ=500*mV # coupling factor

eqs='''
dv/dt=(-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL)+Iapp)/Cm : volt
dm/dt=10*(alpham*(1-m)-betam*m) : 1
alpham=-0.1/mV*(vshift+35*mV)/(exp(-0.1/mV*(vshift+35*mV))-1)/ms : Hz
betam=4*exp(-(vshift+60*mV)/(18*mV))/ms : Hz
vshift=v+KJ*m**3*h : volt
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

run(50*ms,report='text')

subplot(211)
plot(M.times/ms,M[0]/mV)
subplot(212)
plot((M[0]/mV)[:-1],diff(M[0])/defaultclock.dt)
show()
