'''
A pure Python version of the CUBA example, that reproduces basic Brian principles.

Fixed overhead is about 0.35 s.
Real time simulation for N=1300.
'''
from pylab import *
import bisect
from time import time
from random import sample
from scipy import random as scirandom

"""
Parameters
"""
N = 1300        # number of neurons
Ne = int(N * 0.8) # excitatory neurons 
Ni = N - Ne       # inhibitory neurons
mV = ms = 1e-3    # units
dt = 0.1 * ms     # timestep
taum = 20 * ms    # membrane time constant
taue = 5 * ms
taui = 10 * ms
p = 80.0 / N # connection probability (80 synapses per neuron)
Vt = -1 * mV      # threshold = -50+49
Vr = -11 * mV     # reset = -60+49
we = 60 * 0.27 / 10 # excitatory weight
wi = -20 * 4.5 / 10 # inhibitory weight
duration = 1000 * ms

"""
Equations
---------
eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

This is a linear system, so each update corresponds to
multiplying the state matrix by a (3,3) 'update matrix'
"""

# Update matrix
A = array([[exp(-dt / taum), 0, 0],
         [taue / (taum - taue) * (exp(-dt / taum) - exp(-dt / taue)), exp(-dt / taue), 0],
         [taui / (taum - taui) * (exp(-dt / taum) - exp(-dt / taui)), 0, exp(-dt / taui)]]).T

"""
State variables
---------------
P=NeuronGroup(4000,model=eqs,
              threshold=-50*mV,reset=-60*mV)
"""
S = zeros((3, N))

"""
Initialisation
--------------
P.v=-60*mV+10*mV*rand(len(P))
"""
S[0, :] = rand(N) * (Vt - Vr) + Vr # Potential: uniform between reset and threshold

"""
Connectivity matrices
---------------------
Pe=P.subgroup(3200) # excitatory group
Pi=P.subgroup(800)  # inhibitory group
Ce=Connection(Pe,P,'ge',weight=1.62*mV,sparseness=p)
Ci=Connection(Pi,P,'gi',weight=-9*mV,sparseness=p)
"""
We_target = []
We_weight = []
for _ in range(Ne):
    k = scirandom.binomial(N, p, 1)[0]
    target = sample(xrange(N), k)
    target.sort()
    We_target.append(array(target))
    We_weight.append(array([1.62 * mV] * k))
Wi_target = []
Wi_weight = []
for _ in range(Ni):
    k = scirandom.binomial(N, p, 1)[0]
    target = sample(xrange(N), k)
    target.sort()
    Wi_target.append(array(target))
    Wi_weight.append(array([-9 * mV] * k))

"""
Spike monitor
-------------
M=SpikeMonitor(P)

will contain a list of (i,t), where neuron i spiked at time t.
"""
spike_monitor = [] # Empty list of spikes

"""
State monitor
-------------
trace=StateMonitor(P,'v',record=0) # record only neuron 0
"""
trace = [] # Will contain v(t) for each t (for neuron 0)

"""
Simulation
----------
run(duration)
"""
t1 = time()
t = 0 * ms
while t < duration:
    # STATE UPDATES
    S[:] = dot(A, S)

    # Threshold
    all_spikes = (S[0, :] > Vt).nonzero()[0]     # List of neurons that meet threshold condition

    # PROPAGATION OF SPIKES
    # Excitatory neurons
    spikes = all_spikes[0:bisect.bisect_left(all_spikes, Ne)]
    for i in spikes:
        S[1, We_target[i]] += We_weight[i]

    # Inhibitory neurons
    spikes = all_spikes[bisect.bisect_left(all_spikes, Ne):] - Ne
    for i in spikes:
        S[2, Wi_target[i]] += Wi_weight[i]

    # Reset neurons after spiking
    S[0, all_spikes] = Vr                       # Reset membrane potential

    # Spike monitor
    spike_monitor += [(i, t) for i in all_spikes]

    # State monitor
    trace.append(S[0, 0])

    t += dt

t2 = time()
print "Simulated in", t2 - t1, "s"
print len(spike_monitor), "spikes"

"""
Plot
----
subplot(211)
raster_plot(M)
subplot(212)
plot(trace.times/ms,trace[0]/mV)
show()
"""
subplot(211)
i, t = zip(*spike_monitor)
plot(i, t, '.')
subplot(212)
plot(arange(len(trace)) * dt / ms, array(trace) / mV)
show()
