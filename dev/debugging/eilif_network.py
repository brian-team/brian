# Network described in:
# Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
# E. Muller et al, Neural Computation (2007)
from brian import *
import time

#c=Clock(dt=.01*ms)

vth=-57*mV
vreset=-70*mV
cm=289.5*pF
gl=28.95*nS
El=-70*mV
qr=3214*nS
taur=1.97*ms
Er=-70*mV
qs=14.48*nS
taus=10*ms
Es=-70*mV
Ee=0*mV
Ei=-75*mV
qe=2*nS
qi=2*nS
taue=1.5*ms
taui=10*ms
Ne=1000
Ni=250

eqs='''
dv/dt=(gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)+gs*(Es-v)+gr*(Er-v))/cm : volt
dge/dt=-ge/taue : siemens
dgi/dt=-gi/taui : siemens
dgs/dt=-gs/taus : siemens
dgr/dt=-gr/taur : siemens
'''

def eilif_reset(P,spikes):
    P.v_[spikes]=vreset
    P.gs_[spikes]+=qs
    P.gr_[spikes]+=qr

neurons=NeuronGroup(1,model=eqs,threshold=vth,reset=eilif_reset)
neurons.v=vreset
input=PoissonGroup(Ne+Ni,rates=5*Hz)
input_ex=input.subgroup(Ne)
input_inh=input.subgroup(Ni)
syn_ex=Connection(input_ex,neurons,'ge')
syn_ex.connect_full(input_ex,neurons,weight=qe)
syn_inh=Connection(input_inh,neurons,'gi')
syn_inh.connect_full(input_inh,neurons,weight=qi)

trace=StateMonitor(neurons,'v',record=0)
gs=StateMonitor(neurons,'gs',record=0)

start_time=time.time()
run(1000*ms)
print "The simulation took",time.time()-start_time,"s"
subplot(211)
plot(trace.times/ms,trace[0]/mV)
subplot(212)
plot(gs.times/ms,gs[0]/nS)
show()
