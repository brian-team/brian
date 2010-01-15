'''
Polychronization: Computation with Spikes
Eugene M. Izhikevich
Neural Computation, 2006

Note that this version doesn't reproduce Izhikevich's figures from the paper,
which is probably due to the fact that he uses a different STDP rule from the
one below. A future version of this example will be released which reproduces
Izhikevich's STDP rule.
'''
from brian import *
import random

dt = defaultclock.dt = 1*ms

M = 100                # number of synapses per neuron
D = 20*ms              # maximal conduction delay
D_min = 1*ms           # minimal conduction delay
D_i = 1*ms             # inhibitory neuron conduction delay

# excitatory neurons
Ne = 800
a_e = 0.02/ms
d_e = 8*mV/ms
s_e = 6*mV
# inhibitory neurons
Ni = 200
a_i = 0.1/ms
d_i = 2*mV/ms
s_i = -5*mV
# all neurons
b = 0.2/ms
c = -65*mV
sm = 10*mV             # maximal synaptic strength
N = Ne+Ni

thalamic_input_weight = 20*mV

eqs = Equations('''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u   : volt
du/dt = a*(b*v-u)                                : volt/second
a                                                : 1/second
d                                                : volt/second
''')
reset = '''
v = c
u += d
'''
threshold = 30*mV

G = NeuronGroup(N, eqs, threshold=threshold, reset=reset)

# Izhikevich's numerical integration scheme
def Izhikevich_scheme(G):
    G.v += 0.5*dt*((0.04/ms/mV)*G.v**2+(5/ms)*G.v+140*mV/ms-G.u)
    G.v += 0.5*dt*((0.04/ms/mV)*G.v**2+(5/ms)*G.v+140*mV/ms-G.u)
    G.u += dt*G.a*(b*G.v-G.u)
G._state_updater = Izhikevich_scheme

Ge = G.subgroup(Ne)
Gi = G.subgroup(Ni)

G.v = c
G.u = b*c
Ge.a = a_e
Ge.d = d_e
Gi.a = a_i
Gi.d = d_i

@network_operation(clock=EventClock(dt=1*ms))
def random_thalamic_input():
    G.v[randint(N)] += float(thalamic_input_weight)

Ce = Connection(Ge, G, 'v', delay=True, max_delay=D)
Ci = Connection(Gi, Ge, 'v', delay=D_i)

for i in xrange(Ne):
    inds = random.sample(xrange(N), M)
    Ce[i, inds] = s_e*ones(len(inds))
    Ce.delay[i, inds] = rand(len(inds))*(D-D_min)+D_min
for i in xrange(Ni):
    inds = random.sample(xrange(Ne), M)
    Ci[i, inds] = s_i*ones(len(inds))

# Simplified version of Izhikevich's rule 
stdp = ExponentialSTDP(Ce, taup=20*ms, taum=20*ms, Ap=0.1, Am=-0.12,
                       wmax=sm, interactions='nearest')

M = SpikeMonitor(G)

run(1*second, report='stderr')

raster_plot(M)
show()
