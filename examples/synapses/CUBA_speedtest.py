# coding: latin-1
"""
CUBA example with delays.

Connection (no delay): 3.5 s
DelayConnection: 5.7 s
Synapses (with precomputed offsets): 6.6 s # 6.9 s
Synapses with weave: 6.4 s
Synapses with zero delays: 5.2 s
"""

from brian import *
import time
from brian.experimental.synapses import *
#log_level_debug()
#set_global_preferences(useweave=True, usecodegen=False)
set_global_preferences(useweave=False, usecodegen=False)

use_synapses = False

do_callgraph = False
if do_callgraph:
    import pycallgraph
    cg_func = 'synapses'
    def ff(pat):
        def f(call_stack, module_name, class_name, func_name, full_name):
            if not 'brian' in module_name: return False
            for n in call_stack + [full_name]:
                if pat in n:
                    return True
            return False
        return f

import random as pyrand
from numpy.random import seed
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
P.v = Vr + rand(len(P)) * (Vt - Vr)

Pe = P.subgroup(3200)
Pi = P.subgroup(800)
we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight

Ce = Connection(Pe, P, 'ge', weight=we, sparseness=0.02, delay=(1*ms, 2*ms))
Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=0.02, delay=(1*ms, 2*ms))
Ce.compress()
Ci.compress()
if use_synapses:
    set_global_preferences(useweave=True)
    Se = Synapses(Pe, P, model = 'w : 1', pre = 'ge += we')
    Si = Synapses(Pi, P, model = 'w : 1', pre = 'gi += wi')
    set_global_preferences(useweave=False)
    for i in xrange(len(Pe)):
        x = Ce[i, :]
        Se[i, x.ind] = True
        Se.delay[i, :] = Ce.delay[i, :]
        if False:
            # check that it really is producing the same connection structure
            # it seems that it is
            syn = Se.synapses_pre[i].data
            tgts = Se.postsynaptic.data[syn]
            delays = Se.delay.data[syn]
            if not (tgts==x.ind).all():
                print 'tgts!=x.ind'
            if not (delays==asarray(Ce.delay[i, :]/defaultclock.dt, dtype=int)).all():
                print 'delays!=Ce.delay'
    for i in xrange(len(Pi)):
        x = Ci[i, :]
        Si[i, x.ind] = True
        Si.delay[i, :] = Ci.delay[i, :]
    del x, Ce, Ci
    import gc
    gc.collect()

# Record the number of spikes
Me = PopulationSpikeCounter(Pe)
Mi = PopulationSpikeCounter(Pi)
# A population rate monitor
M = PopulationRateMonitor(P)
M0 = RecentStateMonitor(P, 'v', record=[0,1,2,3,4], duration=100*ms)

print "Network construction time:", time.time() - start_time, "seconds"
print len(P), "neurons in the network"
print "Simulation running..."

if use_synapses:
    net = Network(P, Se, Si, Me, Mi, M, M0)
else:
    net = Network(P, Ce, Ci, Me, Mi, M, M0)
net.run(100 * msecond)
#M0.plot()
#show()
#exit()

start_time = time.time()
if do_callgraph:
    pycallgraph.start_trace(filter_func=ff(cg_func))

net.run(1 * second)

duration = time.time() - start_time
if do_callgraph:
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('cuba-callgraph.png')

print "Simulation time:", duration, "seconds"
print Me.nspikes, "excitatory spikes"
print Mi.nspikes, "inhibitory spikes"

#plot(M.times / ms, M.smooth_rate(2 * ms, 'gaussian'))
#show()
