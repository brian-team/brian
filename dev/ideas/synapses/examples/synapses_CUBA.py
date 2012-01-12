# coding: latin-1
"""
CUBA example with delays.

Connection (no delay): 3.5 s
DelayConnection: 5.2 s (9.1 s with dt=0.05 ms)
Synapses: 8 s (14.8 s with dt=0.05 ms)
"""

from brian import *
import time
from dev.ideas.synapses.synapses import *
#log_level_debug()

#defaultclock.dt=0.05*ms

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
    Se = Synapses(Pe, P, model = 'w : 1', pre = 'ge += we', max_delay=2*ms)
    Si = Synapses(Pi, P, model = 'w : 1', pre = 'gi += wi', max_delay=2*ms)
    Se.connect_random(sparseness=0.02)
    Si.connect_random(sparseness=0.02)
    Se.delay_pre[:]=10
    Si.delay_pre[:]=10
    #Se[:,:] = '''rand() < .02'''
    #Si[:,:] = '''rand() < .02'''
    print 'finished adding synapses'
else:
########### OLD CODE
    Ce = Connection(Pe, P, 'ge', weight=we, sparseness=0.02, delay=(1*ms,1*ms))
    Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=0.02, delay=(1*ms,1*ms))
    
P.v = Vr + rand(len(P)) * (Vt - Vr)

# Record the number of spikes
Me = PopulationSpikeCounter(Pe)
Mi = PopulationSpikeCounter(Pi)
# A population rate monitor
M = PopulationRateMonitor(P)

print "Network construction time:", time.time() - start_time, "seconds"
print len(P), "neurons in the network"
print "Simulation running..."
run(1 * msecond)
start_time = time.time()

run(1 * second)

duration = time.time() - start_time
print "Simulation time:", duration, "seconds"
print Me.nspikes, "excitatory spikes"
print Mi.nspikes, "inhibitory spikes"
plot(M.times / ms, M.smooth_rate(2 * ms, 'gaussian'))
show()
