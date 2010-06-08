"""
Brian on multiple processors
"""
from brian import *
from multiprocessing import Queue, Process, Pool
from time import time

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(4000, model=eqs,
              threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)

Ce = Connection(Pe, P, 'ge')
Ci = Connection(Pi, P, 'gi')
Ce.connect_random(Pe, P, 0.02, weight=1.62 * mV)
Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)

M = SpikeMonitor(P)

duration = 1 * second
t = 0 * second
p = Pool(2)
t1 = time()
while t < duration:
    #p=Process(NeuronGroup.update,[Pe,Pi])
    #P.reset()
    Ce.do_propagate()
    Ci.do_propagate()
    M()
    t += .1 * ms
t2 = time()
print t2 - t1
raster_plot(M)
show()
