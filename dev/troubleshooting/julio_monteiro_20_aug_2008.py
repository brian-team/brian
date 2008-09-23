#!/usr/bin/env python
from brian import *
from brian.library.IF import *
from general_stdp import *
from heterog_delay import *

N=3
T=500*ms
max_delay = 20*ms
vm0 = -65*mV
w0 = 0.2*vm0/ms
Vt = -30*mV
input_freq = 20*Hz
input_curr = 20*mV

#STDP
gmax = 0.010
tau_pre = 20*ms
tau_post = 20*ms
dA_pre = gmax*.005
dA_post = -dA_pre*1.05

eqs = Izhikevich(a=0.02/ms, b=0.2/ms)
reset = AdaptiveReset(Vr=-75*mV, b=8.0*nA)
model = Model(model=eqs, threshold=Vt, reset=reset, max_delay=max_delay)

G=N*model
G.vm=vm0
G.w=w0

def nextspike():
    interval = 1/input_freq
    nexttime = interval
    while True:
        yield (0,nexttime)
        nexttime += interval


input = SpikeGeneratorGroup(1, nextspike())
Ic = Connection(input, G)
C1 = Connection(G, G, structure="dense", delay=1*ms)
C2 = Connection(G, G, structure="dense", delay=2*ms)
Ic[0,0] = input_curr
Ic[0,1] = input_curr

#for i in range(len(G)):
#    C1[i]=[0.0]*len(G)
#    C2[i]=[0.0]*len(G)

C1[0,2] = 0.009
C2[1,2] = 0.009


stdp1 = SongAbbottSTDP(C1, gmax=gmax, tau_pre=tau_pre, tau_post=tau_post,
                      dA_pre=dA_pre, dA_post=dA_post)
stdp2 = SongAbbottSTDP(C2, gmax=gmax, tau_pre=tau_pre, tau_post=tau_post,
                      dA_pre=dA_pre, dA_post=dA_post)


SI=SpikeMonitor(input)
S=SpikeMonitor(G)
tr_a=StateMonitor(G,'vm',record=[2])
tr_T=StateMonitor(G,'w',record=[2])
print "Before"
print C1.W
print C2.W


run(T)
subplot(211)
raster_plot(SI,S)
subplot(223)
plot(tr_a.times/ms,tr_a[2])
subplot(224)
plot(tr_T.times/ms,tr_T[2])
print "After"
print C1.W
print C2.W
show()