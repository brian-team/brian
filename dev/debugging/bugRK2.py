from brian import *

eqs='''
dv/dt = (-w-v)/(10*ms) + I : volt # the membrane equation
dw/dt = -w/(30*ms) : volt # the adaptation current

I : volt/second
'''

def myreset(P,spikes):
    P.v[spikes]=0*mV # Faster: P.v_[spikes]=0*mV
    P.w[spikes]+=3*mV # the adaptation variable increases with each spike

IF = NeuronGroup(1,model=eqs,reset=myreset,threshold=20*mV, method='RK')

IF.I = 0*mV/ms

run(100*ms)
