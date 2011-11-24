"""
Naundorf et al. (2006)
Unique features...
Nature.

Doesn't work... typos?
"""
from brian import *

mS=msiemens

v12A=-35*mV
kA=6*mV
invtauA=1/(0.1*ms)
invtauI=1/(0.5*ms)
v12C=80*mV # -80 mV?
kC=4*mV
invtauC=1/(30*ms)
C=1*uF
gL=2*mS
VL=-80*mV
gNa=68.4*mS
VNa=50*mV
I0=0*uA
sigma=12*uA
K=1000
J=3.2*mV
tau=50*ms

eqs="""
dv/dt=(gL*(VL-v)+gNa*o*(VNa-v)+I0+sigma*z)/C : volt
dz/dt=-z/tau+tau**-.5*xi : 1 # noisy current
do/dt=alphaA*(h-o)-(invtauI+betaA)*o : 1 # open channels
dh/dt=alphaC*(1-h)-betaC*(h-o)-invtauI*o : 1 # available channels
# Channel kinetics
alphaA=invtauA/(1+exp(-(v+K*J*o-v12A)/kA)) : Hz
betaA=invtauA/(1+exp((v+K*J*o-v12A)/kA)) : Hz
alphaC=invtauC/(1+exp(-(v-v12C)/kC)) : Hz # I corrected * to /
betaC=invtauC/(1+exp((v-v12C)/kC)) : Hz # idem
"""

neuron=NeuronGroup(1,eqs,threshold=0*mV,reset=VL)
neuron.v=VL
neuron.o=0
neuron.h=1

M=StateMonitor(neuron,'v',record=0)

run(2*second,report='text')

plot(M.times/ms,M[0]/mV)

show()
