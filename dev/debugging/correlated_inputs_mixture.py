'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.

Mixture processes
'''
from brian import *
from brian.correlatedspikes import *
from time import time

N=100
nu=rand(N)*20*Hz
P=rand(N,N)
t1=time()
T=mixture_process(nu=nu,P=P,tauc=10*ms,t=1000*ms)
t2=time()
print t2-t1