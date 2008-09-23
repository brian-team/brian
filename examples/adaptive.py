'''
An adaptive neuron model
'''
from brian import *

PG = PoissonGroup(1,500*Hz)
eqs='''
dv/dt = (-w-v)/(10*ms) : volt # the membrane equation
dw/dt = -w/(30*ms) : volt # the adaptation current
'''
def myreset(P,spikes):
    P.v[spikes]=0*mV # Faster: P.v_[spikes]=0*mV
    P.w[spikes]+=3*mV # the adaptation variable increases with each spike
IF = NeuronGroup(1,model=eqs,reset=myreset,threshold=20*mV)

C = Connection(PG,IF,'v')
C.connect_full(PG,IF,3*mV)

MS = SpikeMonitor(PG,True)
Mv = StateMonitor(IF,'v',record=True)
Mw = StateMonitor(IF,'w',record=True)

run(100*ms)

plot(Mv.times/ms,Mv[0]/mV)
plot(Mw.times/ms,Mw[0]/mV)

show()