'''
Spike propagation with pypar.

I am not sure how broadcasting works. Maybe it is not the better strategy.
Other option: send spikes in ring (n turns to transfer all spikes).

Documentation: http://datamining.anu.edu.au/~ole/pypar/DOC
'''

from brian import *
import pypar

# Identification
myid=pypar.rank() # id of this process
nproc=pypar.size() # number of processors
node=pypar.get_processor_name()

print "I am processor %d of %d on node %s"%(myid, nproc, node)

# Generation of spikes
myspikes=randint(100, size=randint(10))
print "I am sending", myspikes

# Broadcasting
spikes=[[] for _ in range(nproc)]
spikes[myid]=myspikes
nspikes=array([0])
for i in range(nproc):
    if i==myid: # that's me!
        # Create spikes
        nspikes[0]=len(myspikes)
    pypar.broadcast(nspikes, i)
    if i!=myid: # This would be the virtual group
        spikes[i]=zeros(nspikes, dtype=int)
    pypar.broadcast(spikes[i], i)

print "I received", spikes
pypar.finalize()

#if myid == 0:
#    x=2*mV
#    pypar.send(x,1)
#    y=pypar.receive(1)
#    print "I got a big number:",y
#    v=zeros(10)
#    pypar.receive(1,buffer=v) # faster with arrays, does not copy the data
#    print "and now random numbers!"
#    print v
#    print "I am sending spikes!"
#    nspikes=array([3])
#    pypar.broadcast(nspikes,0)
#    spikes=randint(100,size=nspikes)
#    pypar.broadcast(spikes,0)
#else:
#    x=pypar.receive(0)
#    print "I received a message:",x
#    print "and I am doubling it"
#    pypar.send(2*x,0)
#    print "I send an array now"
#    pypar.send(randn(10),0,use_buffer=True)
#    nspikes=array([0])
#    pypar.broadcast(nspikes,0)
#    spikes=zeros(nspikes,dtype=int)
#    pypar.broadcast(spikes,0)
#    print "Broadcast result:"
#    print spikes
#pypar.finalize()
