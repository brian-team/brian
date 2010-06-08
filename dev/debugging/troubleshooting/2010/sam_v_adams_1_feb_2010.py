'''
Note: this doesn't work because you have

dw_ij/dt = c_ij(t)d(t)
c_ij(t) = sum over all pairs s, f_c(t)W(s)

where f_c(s) is an alpha function, and W(s) is the STDP learning function.

The problem is that not only is there a differential equation for w, but
there is also a differential equation for c_ij.

We could solve this by having a new system for STDP which allows you to
specify differential equations at a synaptic level.

Would even that be enough?

It would be relatively easy at least to adapt STDP so that it could do:

1. differential equations and spike time code at the separable neuronal level
2. differential equations and spike time code at the synaptic level

For example, code like:

dw/dt = -w/tau

could be turned into Python code like:

w__tmp = -w/tau
w += dt*w__tmp

where here the symbol 'w' would refer to the flattened array of all synapse
weights. Or it could be exactly solved with code like (for suitable consts):

w[:] = const0*w+const1

More synaptic variables could be introduced with extra copies of the
array. This would all work fine for sparse and dense matrices, and may be more
difficult for dynamic matrices. These would have to be linked so that shape
changes in one are mirrored in shape changes in the other. Perhaps it is worth
having a class that ensures several dynamic ConnectionMatrix objects keep the
same underlying structure of nonzero entries? It would be relatively easy to
achieve, it would just be a proxy to the other matrices.

This all works fine for zero delays, how about for heterogeneous delays?
In terms of memory use, it's fine to store a copy of the recent values of the
per-neuron variables, but unrealistic to store a copy of the recent values of
per-synapse variables in all but a very few cases. 
'''

from brian import *

sim_clock = Clock(dt=.1 * ms)
weight_update_clock = Clock(dt=10 * ms)

N = 100
tau_m = 10 * ms

G = NeuronGroup(N, 'dV/dt=-V/tau_m:1', reset=0, threshold=1, clock=sim_clock)

C = Connection(G, G, 'V', delay=True)
C_trace = Connection(G, G, 'V', delay=True)
forget(C_trace)

for i in range(N):
    C_trace[i, :] = C[i, :]
    C_trace.delay[i, :] = C.delay[i, :]

C_trace.compress()
C_trace.W.alldata[:] = 0

@network_operation(clock=weight_update_clock)
def weight_update():
    C.W.alldata += weight_update_clock.dt * C_trace.W.alldata

stdp = STDP(C_trace, '''
            dA_pre/dt  = (A_pre_aux-A_pre)/tau_pre : 1
            dA_pre_aux/dt = -A_pre_aux/tau_pre_aux : 1
            dA_post/dt = (A_post_aux-A_post)/tau_post : 1
            dA_post_aux/dt = -A_post_aux/tau_post_aux : 1
            ''', pre='''
            A_pre_aux += delta_A_pre_aux
            w += A_post
            ''', post='''
            A_post_aux += delta_A_post_aux
            w += A_pre
            ''', wmax=wmax)
