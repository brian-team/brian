import brian_no_units
from brian import *
import time

#set_global_preferences(useweave=False)
print 'Compilation on', get_global_preference('useweave')

duration=1*second
Nsyn=80.
we=1.62*mV
N=512*16
structure='dense'

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

Ne=int(N*0.8)
Ni=N-Ne

P=NeuronGroup(N, model=eqs,
              threshold=-50*mV, reset=-60*mV, method='Euler', compile=True, freeze=True)#, refractory=5*ms)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)

Ce=Connection(Pe, P, 'ge', structure=structure)
Ci=Connection(Pi, P, 'gi', structure=structure)
cp=Nsyn/N
Ce.connect_random(Pe, P, cp, weight=we)
Ci.connect_random(Pi, P, cp, weight=-9*mV)

M=PopulationSpikeCounter(P)

P.v=-60*mV+10*mV*rand(len(P))

#Ce.W.freeze()
#Ci.W.freeze()

run(0*ms)

def f():
    tstart=time.time()
    run(duration)
    tend=time.time()

    print 'N:', N
    print 'CPU time:', tend-tstart
    print 'Num spikes', M.nspikes

f()
#import cProfile as profile
#import pstats
#profile.run('f()','cuba_once.prof')
#stats = pstats.Stats('cuba_once.prof')
##stats.strip_dirs()
#stats.sort_stats('cumulative', 'calls')
#stats.print_stats(50)
