'''
Very short example program.
'''

from numpy import *
#from pylab import *
import time
import random

second=1.
ms=0.001
mV=0.001

N=4000
dt=0.1*ms
T=100*ms
numsteps=int(T/dt)
Vr=-60*mV
Vt=-50*mV

S=zeros((3, N))
A=array([[ 0.99501248, 0.00493794, 0.00496265],
 [ 0.     , 0.98019867, 0.        ],
 [ 0.  , 0.  , 0.99004983]])
_C=array([[ -2.44388520e-04],
 [ -8.58745657e-21],
 [  6.90431479e-20]])

S[0, :]=[random.uniform(Vr, Vt) for _ in xrange(N)]

l=[[] for _ in range(N)]

t=0
start=time.time()
Nspike=0
for _ in xrange(numsteps):
    S[:]=dot(A, S)+_C
    I=where(S[0, :]>Vt)[0]
    Nspike+=len(I)
    S[0, I]=Vr
    t+=dt
#    for i in range(N):
#        l[i].append(S[0,i])
print time.time()-start
print Nspike

#for i in range(N):
#    plot(l[i])
#show()

#from brian import *
#
#eqs='''
#dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
#dge/dt = -ge/(5*ms) : volt
#dgi/dt = -gi/(10*ms) : volt
#'''
#
#P=NeuronGroup(4000,model=eqs,
#              threshold=-50*mV,reset=-60*mV)

#P.v=-60*mV+10*mV*rand(len(P))
#Pe=P.subgroup(3200)
#Pi=P.subgroup(800)
#
#Ce=Connection(Pe,P,'ge')
#Ci=Connection(Pi,P,'gi')
#Ce.connect_random(Pe, P, 0.02,weight=1.62*mV)
#Ci.connect_random(Pi, P, 0.02,weight=-9*mV)
#
#M=SpikeMonitor(P)
#
#run(1*second)
#raster_plot(M)
#show()
