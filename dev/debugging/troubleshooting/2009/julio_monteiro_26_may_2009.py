#!/usr/bin/env python
from brian import *
from brian.library.IF import *

T=3000*ms

Ni=200
Ne=800
N=Ni+Ne
conn_prob=0.1
exc_weigth0=6*mV
inh_weigth0=-5*mV
max_delay=20*ms
vm0=-65*mV
w0=0.2*vm0/ms
Vt=-30*mV
input_freq=20*Hz
input_curr=20*mV

eqs_e=Izhikevich(a=0.02/ms, b=0.2/ms)
rst_e=AdaptiveReset(Vr=-75*mV, b=8.0*nA)
Me=Model(model=eqs_e, threshold=Vt, reset=rst_e, max_delay=max_delay)

eqs_i=Izhikevich(a=0.1/ms, b=0.2/ms)
rst_i=AdaptiveReset(Vr=-75*mV, b=2.0*nA)
Mi=Model(model=eqs_i, threshold=Vt, reset=rst_i, max_delay=max_delay)

Gi=Ni*Mi
Gi.vm=vm0
Gi.w=w0

Ge=Ni*Mi
Ge.vm=vm0
Ge.w=w0

def nextspike():
    interval=1/input_freq
    nexttime=interval
    while True:
        yield (0, nexttime)
        nexttime+=interval

# random thalamic input
input=SpikeGeneratorGroup(1, nextspike())
Ice=Connection(input, Ge)
Ici=Connection(input, Gi)

# uniform distribution of exc. synaptic delays
Cee=DelayConnection(Ge, Ge, max_delay=20*ms)
Cee.connect_random(Ge, Ge, conn_prob/2, exc_weigth0, delay=(0*ms, 20*ms))
Cei=DelayConnection(Ge, Gi, max_delay=20*ms)
Cei.connect_random(Ge, Gi, conn_prob/2, exc_weigth0, delay=(0*ms, 20*ms))

Cie=DelayConnection(Gi, Ge, max_delay=1*ms)
# all inhibitory delays are 1 ms
Cie.connect_random(Gi, Ge, conn_prob, inh_weigth0, delay=1*ms)

#STDP
gmax=0.010
tau_pre=20*ms
tau_post=20*ms
dA_pre=gmax*.005
dA_post=-dA_pre*1.05

eqs_stdp="""
dA_pre/dt  = -A_pre/tau_pre   : 1
dA_post/dt = -A_post/tau_post : 1
"""
stdp_ee=STDP(Cee, eqs=eqs_stdp, pre='''A_pre+=dA_pre; w+=A_post''',
            post='''A_post+=dA_post; w+=A_pre''', wmax=gmax)
stdp_ei=STDP(Cei, eqs=eqs_stdp, pre='''A_pre+=dA_pre; w+=A_post''',
            post='''A_post+=dA_post; w+=A_pre''', wmax=gmax)
stdp_ie=STDP(Cie, eqs=eqs_stdp, pre='''A_pre+=dA_pre; w+=A_post''',
            post='''A_post+=dA_post; w+=A_pre''', wmax=gmax)

Se=SpikeMonitor(Ge)
Si=SpikeMonitor(Gi)

run(T)
raster_plot(Se, Si)

show()
