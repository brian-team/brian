'''
Fast implementation of Brette-Gerstner model

Idea:
u=gl*deltat*exp((v-vt)/deltat)
du/dt=(u/deltat)*dv/dt
'''
from brian import *
from time import time

defaultclock.dt=.01*ms

N=10
C=281*pF # Can be fixed
gL=30*nS
taum=C/gL
EL=-70.6*mV # Same as changing I
VT=-50.4*mV
DeltaT=2*mV
Vcut=VT+5*DeltaT
tauw=144*ms
a=4*nS
b=0.0805*nA
Vr=-70.6*mV
I=.6*nA

eqs=Equations("""
dvm/dt=(gL*(EL-vm)+u+I-w)/C : volt
dw/dt=(a*(vm-EL)-w)/tauw : amp
du/dt=(u/DeltaT)*(gL*(EL-vm)+u+I-w)/C : amp
I : amp
""")

def myreset(P,spikes):
    P.vm[spikes]=Vr
    P.w[spikes]+=b
    P.u[spikes]=gL*DeltaT*exp((Vr-VT)/DeltaT)

neuron=NeuronGroup(N,model=eqs,threshold=Vcut,reset=myreset,freeze=True)
neuron.vm=EL
neuron.w=0*amp
neuron.u=gL*DeltaT*exp((Vr-VT)/DeltaT)
M=StateMonitor(neuron,'vm',record=[0])

t1=time()
run(200*ms)
t2=time()
print t2-t1
plot(M.times/ms,M[0]/mV)
show()
