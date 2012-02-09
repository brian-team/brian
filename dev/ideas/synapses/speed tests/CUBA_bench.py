
from brian import *
import time
from brian.experimental.synapses import *
    #log_level_debug()
set_global_preferences(useweave = True)

import random as pyrand
from numpy.random import seed


def one_run(use_synapses = True):
    # coding: latin-1
    """
    CUBA example with delays.

    Connection (no delay): 3.5 s
    DelayConnection: 5.7 s
    Synapses (with precomputed offsets): 6.6 s # 6.9 s
    Synapses with weave: 6.4 s
    Synapses with zero delays: 5.2 s
    """


    clear(True, True)
    reinit_default_clock()
    
    seed(3240832)
    pyrand.seed(324331)
    
    

    start_time = time.time()
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV

    eqs = Equations('''
    dv/dt  = (ge+gi-(v-El))/taum : volt
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    ''')

    P = NeuronGroup(4000, model=eqs, threshold=Vt, reset=Vr, refractory=5 * ms)
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV

    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight

    if use_synapses:
    ########### NEW SYNAPSE CODE
        Se = Synapses(Pe, P, model = 'w : 1', pre = 'ge += we')
        Si = Synapses(Pi, P, model = 'w : 1', pre = 'gi += wi')
        Se[:,:]=0.02
        Si[:,:]=0.02
        Se.delay='rand()*ms'
        Si.delay='rand()*ms'
    else:
    ########### OLD CODE
        Ce = Connection(Pe, P, 'ge', weight=we, sparseness=0.02, delay=(0*ms,1*ms))
        Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=0.02, delay=(0*ms,1*ms))

    P.v = Vr + rand(len(P)) * (Vt - Vr)

    # Record the number of spikes
    Me = PopulationSpikeCounter(Pe)
    Mi = PopulationSpikeCounter(Pi)
    # A population rate monitor
    M = PopulationRateMonitor(P)


    run(1 * msecond)
    start_time = time.time()

    run(1 * second)

    duration = time.time() - start_time

    return duration
    
if __name__ == '__main__':
    Niter = 5
    perf_synapses = np.zeros(Niter)
    perf_nosynapses = np.zeros(Niter)
    for k in range(Niter):
        print k
        perf_synapses[k] = one_run()
#        perf_nosynapses[k] = one_run(use_synapses = False)
    
    from matplotlib.pyplot import *
    hist([perf_synapses, perf_nosynapses])
    show()
    
