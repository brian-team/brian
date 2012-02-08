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

Pe = P.subgroup(3200)
Pi = P.subgroup(800)
we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight

if True:
########### NEW SYNAPSE CODE
    set_global_preferences(useweave=True)
    Se = Synapses(Pe, P, model = 'w : 1', pre = 'ge += we')
    Si = Synapses(Pi, P, model = 'w : 1', pre = 'gi += wi')
    set_global_preferences(useweave=False)
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

print "Network construction time:", time.time() - start_time, "seconds"
print len(P), "neurons in the network"
print "Simulation running..."

net = MagicNetwork()
net.run(100 * msecond)
#run(100 * msecond)

start_time = time.time()
if do_callgraph:
    pycallgraph.start_trace(filter_func=ff(cg_func))

net.run(1 * second)
#run(1 * second)

duration = time.time() - start_time
if do_callgraph:
    pycallgraph.stop_trace()
    pycallgraph.make_dot_graph('cuba-callgraph.png')

print "Simulation time:", duration, "seconds"
print Me.nspikes, "excitatory spikes"
print Mi.nspikes, "inhibitory spikes"

#plot(M.times / ms, M.smooth_rate(2 * ms, 'gaussian'))
#show()
