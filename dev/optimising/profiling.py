'''
Very short example program.

Profiling:
* state update = 0.17 s, including 0.11 s in state updater
  but only 0.016 if no network update
* threshold = 0.22 s
* reset = 0.06 s
* each connection = about 0.20 s (even with no spike)
'''
#import brian_no_units
from brian import *
from time import time

N = 5        # number of neurons
Ne = int(N * 0.8) # excitatory neurons 
Ni = N - Ne       # inhibitory neurons
p = .1
duration = 1000 * ms

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(N, model=eqs)#,threshold=-50*mV,reset=-60*mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
#P._state_updater=lambda _:0
#Pe=P.subgroup(Ne)
#Pi=P.subgroup(Ni)

#Ce=Connection(Pe,P,'ge',weight=0*mV,sparseness=p)
#Ci=Connection(Pi,P,'gi',weight=0*mV,sparseness=p)

run(1 * ms)
t1 = time()
run(duration)
t2 = time()
print "Simulated in", t2 - t1, "s"
