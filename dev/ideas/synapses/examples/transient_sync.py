'''
Transient synchronisation in a population of noisy IF neurons
with distance-dependent synaptic weights (organised as a ring)
'''
from brian import *
import time
from dev.ideas.synapses.synapses import *

tau = 10 * ms
N = 100
v0 = 5 * mV
sigma = 4 * mV
group = NeuronGroup(N, model='dv/dt=(v0-v)/tau + sigma*xi/tau**.5 : volt', \
                  threshold=10 * mV, reset=0 * mV)
C = Synapses(group,model='w:1',pre='v+=w')
C[:,:]=True
C.w='.4 * mV * cos(2. * pi * (i - j) * 1. / N)'
S = SpikeMonitor(group)
R = PopulationRateMonitor(group)
group.v = rand(N) * 10 * mV

run(5000 * ms,report='text')
subplot(211)
raster_plot(S)
subplot(223)
imshow(C.w[:].reshape((N,N)), interpolation='nearest')
title('Synaptic connections')
subplot(224)
plot(R.times / ms, R.smooth_rate(2 * ms, filter='flat'))
title('Firing rate')
show()