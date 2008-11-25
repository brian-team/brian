import brian_no_units
from brian import *
import time

#set_global_preferences(useweave=False)
print 'Compilation on', get_global_preference('useweave')

duration = 2.5*second
Nsyn = 80.
we = 9*mV
N = 16000

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

Ne = int(N*0.8)
Ni = N-Ne

P=NeuronGroup(N,model=eqs,
              threshold=-50*mV,reset=-60*mV, refractory=5*ms)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)

Ce=Connection(Pe,P,'ge')
Ci=Connection(Pi,P,'gi')    
cp = Nsyn/N    
Ce.connect_random(Pe, P, cp, weight=we)
Ci.connect_random(Pi, P, cp, weight=-9*mV)

M = PopulationSpikeCounter(P)

P.v = -60*mV+10*mV*rand(len(P))

#Ce.W.freeze()
#Ci.W.freeze()

run(0*ms)

def f():
    tstart = time.time()
    run(duration)
    tend = time.time()
    
    print 'Time taken', tend-tstart
    print 'Num spikes', M.nspikes

f()
#import cProfile as profile
#import pstats
#profile.run('f()','cuba_once.prof')
#stats = pstats.Stats('cuba_once.prof')
##stats.strip_dirs()
#stats.sort_stats('cumulative', 'calls')
#stats.print_stats(50)
