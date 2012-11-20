#!/usr/bin/env python
'''
CUBA network
'''
from pylab import *
from brian2 import *
import time

start = time.time()

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(4000, eqs,
                threshold='v>-50*mV', reset='v=-60*mV',
                #language=CPPLanguage(),
                )
P.v = -60*mV+10*mV*rand(len(P))
#Pe = P.subgroup(3200)
#Pi = P.subgroup(800)
#
#Ce = Connection(Pe, P, 'ge', weight=1.62 * mV, sparseness=0.02)
#Ci = Connection(Pi, P, 'gi', weight= -9 * mV, sparseness=0.02)

M = SpikeMonitor(P)

print 'Initialising objects:', time.time()-start
start = time.time()
run(0.1*ms)
print 'Initialising network run:', time.time()-start
start = time.time()
run(1*second-0.1*ms)
print 'Simulation time:', time.time()-start
start = time.time()
i, t = M.it
print 'Spike monitor variable extraction time:', time.time()-start
start = time.time()
plot(t, i, '.k')
print 'Plotting time (matplotlib):', time.time()-start
print
print 'Number of spikes:', M.num_spikes
show()
