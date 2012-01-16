"""
Speed test for Synapses
-----------------------
Results look ok.

With M = 10
740 000 spikes/second
No synapses: 0.5s
Only the spike queue: 6.5s (46%)
Complete: 13.5s (54%) (but this is a case with many simultaneous updates on the same post neuron)

With M = 1
311 000 spikes/second
No synapses: 0.5s
Only the spike queue: 2.4s (70%)
Complete: 3.2s (30%)

Thus in practical cases, the spike queue is probably the dominant computational cost.
With a C implementation spike queue, we will then probably get delay connections that
are as fast as normal connections.

For comparison:
CUBA:320000 synapses, about 5 Hz -> 1,600,000 events / biological second
non-spiking CUBA: 1.5s
CUBA with Connection: 3s
CUBA with DelayConnection: 5.3s
    -> about 4s for spikes -> 400,000 events/second

Nemo:  400 000 000 spikes/second
Jayram: 80 000 000 spikes/second
"""
from brian import *
from brian.experimental.synapses import *
from time import time

#log_level_debug()

N=100
M=10
rate=10000*Hz
duration=1*second
P=NeuronGroup(1,model='dv/dt=rate :1',threshold=1,reset=0)
Q=NeuronGroup(N,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w',max_delay=2*ms)

#S[0,:]=True
S[P,Q]=M
S.w=rand(N*M)
S.delay_pre[:]=rand(N*M)*1*ms

#S.pre_queue.precompute_offsets()
run(1*ms)
t1=time()
run(duration)
t2=time()
print "Simulation time:",t2-t1,"s"
print rate*duration*N*M,"events"
print rate*duration*N*M/(t2-t1),"events/s"
