'''
Polychronization: Computation with Spikes
Eugene M. Izhikevich
Neural Computation, 2006

This script shows a way to reproduce Izhikevich's network using Brian. Note
that there are a couple of slightly tricky technical points necessary to
reproduce Izhikevich's particular numerical integration scheme and STDP rule.
These are explained below.

This version of the script demonstrates the simplest version of the code
possible, the longer version of this script shows a much faster version which
saves the output and analyses the polychronous groups formed.
'''

# This is to use interactive plotting
import matplotlib
matplotlib.use('WXAgg') # You may need to experiment, try WXAgg, GTKAgg, QTAgg, TkAgg

from brian import *
import random

dt = defaultclock.dt = .5 * ms

I_factor = float((1*ms)/dt)

M = 100                # number of synapses per neuron
D = 20 * ms              # maximal conduction delay
D_min = 1 * ms           # minimal conduction delay
D_i = 1 * ms             # inhibitory neuron conduction delay

# excitatory neurons
Ne = 800
a_e = 0.02 / ms
d_e = 8 * mV / ms
s_e = 6 * mV / ms * I_factor
# inhibitory neurons
Ni = 200
a_i = 0.1 / ms
d_i = 2 * mV / ms
s_i = -5 * mV / ms * I_factor
# all neurons
b = 0.2 / ms
c = -65 * mV
sm = 10 * mV / ms * I_factor           # maximal synaptic strength
N = Ne + Ni

thalamic_input_weight = 20 * mV / ms# * I_factor

eqs = Equations('''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u+I : volt
du/dt = a*(b*v-u)                                : volt/second
a                                                : 1/second
d                                                : volt/second
I                                                : volt/second
''')
reset = '''
v = c
u += d
'''
threshold = 30 * mV

G = NeuronGroup(N, eqs, threshold=threshold, reset=reset)

# Izhikevich's numerical integration scheme is not built in to Brian, but
# we can specify it explicitly. By setting the NeuronGroup._state_updater
# attribute to a function f(G), we replace the default numerical integration
# scheme (Euler) defined by Brian. This function can run arbitrary Python
# code.
def Izhikevich_scheme(G):
    G.v += 0.5 * dt * ((0.04 / ms / mV) * G.v ** 2 + (5 / ms) * G.v + 140 * mV / ms - G.u + G.I)
    G.v += 0.5 * dt * ((0.04 / ms / mV) * G.v ** 2 + (5 / ms) * G.v + 140 * mV / ms - G.u + G.I)
    G.u += dt * G.a * (b * G.v - G.u)
G._state_updater = Izhikevich_scheme

Ge = G.subgroup(Ne)
Gi = G.subgroup(Ni)

G.v = c
G.u = b * c
Ge.a = a_e
Ge.d = d_e
Gi.a = a_i
Gi.d = d_i

# In Izhikevich's simulation, the value of I is set to zero each time step,
# then a 'thalamic input' is added to I, then spikes cause a postsynaptic
# effect on I, and then the numerical integration step is performed. In
# Brian, the numerical integration step is performed before spike propagation,
# so we insert the thalamic input after the numerical integration but before
# the spike propagation, using the when='before_connections' setting.

change_every = int(1*ms/dt)
change_time = 0
thal_ind = 0
@network_operation(when='before_connections')
def thalamic_input():
    global change_time, thal_ind
    G.I = 0 * mV / ms
    G.I[thal_ind] = float(thalamic_input_weight)
    if change_time==0:
        change_time = change_every
        thal_ind = randint(N)
        #G.I[randint(N)] = float(thalamic_input_weight*1*ms/dt)
    change_time -= 1

Ce = Connection(Ge, G, 'I', delay=True, max_delay=D)
Ci = Connection(Gi, Ge, 'I', delay=D_i)

# In order to implement Izhikevich's STDP rule, we need two weight matrices,
# one is the actual weight matrix and the second is a weight matrix derivative.
# To do this with Brian's sparse matrices, we create a second Connection
# Ce_deriv, which we initialise to have the same shape as Ce, but we use the
# forget() function on the Connection so that although the object exists,
# no spikes will be propagated by that Connection. We need the object to exist
# because we create an ExponentialSTDP object that acts on Ce_deriv not
# directly on Ce.

Ce_deriv = Connection(Ge, G, 'I', delay=True, max_delay=D)
forget(Ce_deriv)

for i in xrange(Ne):
    inds = random.sample(xrange(N), M)
    Ce[i, inds] = s_e * ones(len(inds))
    Ce.delay[i, inds] = rand(len(inds)) * (D - D_min) + D_min
for i in xrange(Ni):
    inds = random.sample(xrange(Ne), M)
    Ci[i, inds] = s_i * ones(len(inds))

# Now we duplicate Ce to Ce_deriv

for i in xrange(Ne):
    Ce_deriv[i, :] = Ce[i, :]
    Ce_deriv.delay[i, :] = Ce.delay[i, :]

# STDP acts directly on Ce_deriv rather than Ce. With this STDP rule alone,
# we would not see any learning, the network operation below propagates changes
# in Ce_deriv to Ce.

artificial_wmax = 1e10 * mV / ms
stdp = ExponentialSTDP(Ce_deriv,
                       taup=20 * ms, taum=20 * ms,
                       Ap=(0.1 * mV / ms / artificial_wmax),
                       Am=(-0.12 * mV / ms / artificial_wmax),
                       wmin= -artificial_wmax,
                       wmax=artificial_wmax,
                       interactions='nearest'
                       )

# Izhikevich's STDP rule has STDP acting on a matrix sd of derivatives, and
# then each second the weight matrix s and derivates sd are updated according
# to the rule:
#   s <- s+0.01+sd
#   sd <- sd*0.9
# Note also that we are using Brian's sparse matrices, but they are have
# exactly the same nonzero entries, and therefore we can do arithmetic
# operations on these matrices using the alldata attribute of Brian sparse
# matrices. The compress() method converts the ConstructionMatrix into a
# ConnectionMatrix, thus freezing the nonzero entries. In the next line, we
# want the actual starting values of Ce_deriv to be zero, but if we had done
# this before compressing to a ConnectionMatrix, all entries would be
# considered zero entries in the sparse matrix, and then Ce and Ce_deriv would
# have a different pattern of non-zero entries.

Ce_deriv.compress()
Ce_deriv.W.alldata[:] = 0
@network_operation(clock=EventClock(dt=1 * second))
def update_weights_from_derivative():
    Ce.W.alldata[:] = clip(Ce.W.alldata + 0.01 * mV / ms + Ce_deriv.W.alldata, 0, sm)
    Ce_deriv.W.alldata[:] *= 0.9

M = SpikeMonitor(G)

ion()
raster_plot(M, refresh=.2 * second, showlast=1 * second)

run(50 * second, report='stderr')

ioff()
show()
