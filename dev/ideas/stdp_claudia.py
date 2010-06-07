# -*- coding:utf-8 -*-
"""
Voltage-dependent STDP from:
Connectivity reflects coding: a model of voltage-based STDP with homeostasis.
Clopath C, Büsing L, Vasilaki E, Gerstner W.
Nat Neurosci. 2010 Mar;13(3):344-52
"""

from brian import *

tau_x=10*ms
tau_minus=10*ms
tau_plus=10*ms
delta_x=1.
theta=-65*mV
A_LTD=-0.1
A_LTP=0.1
delta_u_minus=5*mV
delta_u_plus=5*mV

eqs_stdp="""
dx/dt=-x/tau_x : 1                      # presynaptic trace
du_minus/dt=(v-u_minus)/tau_minus : volt   # postsynaptic trace 1 
du_plus/dt=(v-u_plus)/tau_plus : volt      # postsynaptic trace 2
v : volt
"""

pre="""
x+=delta_x
w+=clip(u_minus-theta,0,inf)*A_LTD  # A_LTD<0; theta is theta_minus
"""

post="""
u_minus+=delta_u_minus
u_plus+=delta_u_plus
w+=clip(u_plus-theta,0,inf)*x*A_LTP  # A_LTP>0
"""

# The model has hard bounds wmin and wmax
# NB: maybe we should have a rectify function (=clip(x,0,inf))

N=NeuronGroup(2, 'dv/dt=1*Hz:1')
C=Connection(N, N, 'v')
mystdp=STDP(C, eqs=eqs_stdp, pre=pre, post=post, wmax=2*mV)
mystdp.post_group.v=linked_var(N, 'v')

run(1*ms)
