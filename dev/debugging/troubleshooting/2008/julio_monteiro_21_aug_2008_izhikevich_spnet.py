#!/usr/bin/env python
#import brian_no_units
from brian import *
from brian.library.IF import *
from brian.experimental.general_stdp import *
from brian.experimental.heterog_delay import *
from brian.experimental.stdp_sparse import *
from brian.connection import DenseConnectionMatrix
import numpy
import time

#set_global_preferences(useweave=True)

defaultclock.dt = 0.5 * ms

class VariableIzhikevichReset(object):
    def __call__(self, P):
        spikes = P.LS.lastspikes()
        P.vm_[spikes] = Vr
        P.w_[spikes] += P.d_[spikes]

T = 10000 * ms

#Network parameters
N = 1000 # total number of neurons
Ne = 800 # number of excitatory neurons
Ni = 200 # number of inhibitory neurons
Nc = 100 # number of synapses per neuron

vm0 = -65 * mV # the initial membrane current
w0 = 0.2 * vm0 / ms # the initial recovery varible
Vr = -65 * mV
Vt = 30 * mV # "firing" threshold
min_delay = 1 * ms
max_delay = 20 * ms # maximum axonal delay
i_delay = 1 * ms # the inhibitory delay
wE0 = 6 * mV #initial positive synaptic weight
wI0 = -5 * mV #initial negative synaptic weight
# I think these are wrong, the units should be mV
#wE0= 6*nA #initial positive synaptic weight
#wI0= -5*nA #initial negative synaptic weight

#STDP
gmax = 10 * mV#0.010 # maximum synaptic weight
tau_pre = 20 * ms
tau_post = 20 * ms
dA_pre = gmax * .005
dA_post = -dA_pre * 1.05

eqs = Izhikevich(a='a', b=0.2 / ms) + '''
        a : Hz
        d : volt/second
        '''
model = Model(model=eqs, threshold=Vt, reset=VariableIzhikevichReset())

#dt bit needed for versions of Brian which aren't recent
G = NeuronGroup(N, model, max_delay=i_delay + defaultclock.dt, compile=True, freeze=True)

Ge = G.subgroup(Ne)
Gi = G.subgroup(Ni)

Ge.vm = vm0
Ge.w = w0
Ge.a = 0.02 / ms
Ge.d = 8.0 * mV / ms

Gi.vm = vm0
Gi.w = w0
Gi.a = 0.1 / ms
Gi.d = 2.0 * mV / ms

start_time = time.time()
Ce = HeterogeneousDelayConnection(Ge, G, max_delay=max_delay, delays_per_synapse=True)
#Cei = HeterogeneousDelayConnection(Ge, Gi, max_delay=max_delay, delays_per_synapse=True)
Cie = Connection(Gi, Ge, delay=1 * ms)

# Add Nc connections to each neuron in the network
#Ce.connect_random(Ge, Ge, p=float(Nc)/N, weight=wE0)
#Ce.connect_random(Ge, Gi, p=float(Nc)/N, weight=wI0)#really want this to be wI0?
Ce.connect_random(Ge, G, p=float(Nc) / N, weight=wE0)
for i in range(len(Ge)):
    Ce[i, i] = 0
Ce.delayvec = DenseConnectionMatrix((len(Ge), len(G))) # faster in the current implementaton I think

# This way is slow, but if we're making delayvec a sparse matrix we have to do it like this
#for i in xrange(len(Ge)):
#    for j in xrange(len(G)):
#        if abs(Ce[i,j])>1e-30:
#            Ce.delayvec[i,j]=rand()*(max_delay-min_delay)+min_delay

Ce.delayvec[:] = random((len(Ge), len(G))) * (max_delay - min_delay) + min_delay
# Use this if you need to be sure they are integer multiples of 1ms, but not necessary I think
#Ce.delayvec[:] = ((random((len(Ge), len(G)))*max_delay/ms).astype(int)).astype(float)*ms

# An alternative idea: SparseSTDPConnectionMatrix faster for delayvec too? Answer: not noticeably
#Ce.delayvec = SparseSTDPConnectionMatrix(Ce.W)
#Ce.delayvec[:] = rand(len(Ce.delayvec[:]))*(max_delay-min_delay)+min_delay
#Ce.delayvec.ndim = 2

Cie.connect_random(Gi, Ge, p=float(Nc) / N, weight=wI0)

# if you turn STDP off it takes 20 times longer! The SparseSTDPConnectionMatrix is MORE efficient than
# what was there before!!
STDPe = SongAbbottSTDP(Ce, gmax=gmax, tau_pre=tau_pre, tau_post=tau_post,
                      dA_pre=dA_pre, dA_post=dA_post)
Ce.W = SparseSTDPConnectionMatrix(Ce.W)

print "Network build time:", time.time() - start_time, "seconds"

thalamic_alternate = True

Gvm_ = G.vm_
thaldelta = float(20 * mV)

@network_operation(when='start')
def thalamic_input():
    global thalamic_alternate
    # at each cycle, some random neuron gets a 20*mV boost
    if thalamic_alternate:
        n = int(N * random())
        Gvm_[n] += thaldelta#float(20*mV)
        thalamic_alternate = False
    else:
        thalamic_alternate = True

S = SpikeMonitor(G)

net = MagicNetwork()
#net.update_schedule_groups_resets_connections()

def do_run(T=T, doplot=True):
    start_time = time.time()
    net.run(T)
    print "Excecution time:", time.time() - start_time, "seconds"
    if doplot:
        raster_plot(S)
        show()

do_run()

#import cProfile
#cProfile.run('do_run(1000*ms,False)')

#import pycallgraph
#cg_func = 'Network.update'
#def ff(pat):
#    def f(call_stack, module_name, class_name, func_name, full_name):
#        if not 'brian' in module_name: return False
#        for n in call_stack+[full_name]:
#            if pat in n:
#                return True
#        return False
#    return f
#pycallgraph.start_trace(filter_func=ff(cg_func))
#do_run(1*second,False)
#pycallgraph.stop_trace()
#pycallgraph.make_dot_graph('callgraph-'+cg_func+'.png')
