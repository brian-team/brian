'''
Very short example program.
'''
import brian_no_units
from brian import *
import time
import numpy

set_global_preferences(useweave=True)

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P=NeuronGroup(4000,model=eqs,
              threshold=-50*mV,reset=-60*mV)
numpy.random.seed(593240439)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(3200)
Pi=P.subgroup(800)

Ce=Connection(Pe,P,'ge')#, structure='dense')
Ci=Connection(Pi,P,'gi')#, structure='dense')
Ce.connect_random(Pe, P, 0.02,weight=1.62*mV, seed=23483024)
Ci.connect_random(Pi, P, 0.02,weight=-9*mV, seed=34238035)

#M=SpikeMonitor(P)
M=PopulationSpikeCounter(P)

start = time.time()
run(1*second)
print time.time()-start
print M.nspikes

#raster_plot(M)
#show()