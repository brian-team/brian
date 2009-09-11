from brian import *
from time import time

m=2
k=10000 # number of time steps

class fake(object):
    pass

if m==1:
    t1=time()
    N=10000000
    for _ in xrange(N):
        x=5.
    t2=time()
    print k*(t2-t1)*second/N
elif m==2:
    t1=time()
    N=100000
    for _ in xrange(N):
        #x=1*second
        #x=second.copy()
        #x=Quantity(2.)
        #x=fake()
        #x.dim=D
        x=Quantity.with_dimensions(2.,second.dim)
    t2=time()
    print k*(t2-t1)*second/N
