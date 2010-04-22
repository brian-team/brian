# -*- coding:utf-8 -*-
"""
Voltage-dependent STDP from:
Connectivity reflects coding: a model of voltage-based STDP with homeostasis.
Clopath C, Büsing L, Vasilaki E, Gerstner W.
Nat Neurosci. 2010 Mar;13(3):344-52
"""

eqs_stdp="""
dx/dt=-x/tau_x : 1                      # presynaptic trace
du_minus/dt=(v-u_minus)/tau_minus : 1   # postsynaptic trace 1 
du_plus/dt=(v-u_plus)/tau_plus : 1      # postsynaptic trace 2
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
