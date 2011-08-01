"""
Spike queues following BEP-21
"""
from brian import *
from time import time

# This is a 2D circular array
class SpikeQueue(object):
    '''
    A spike queue, implemented as a circular 2D array.
    
    * Initialize with the number of timesteps and the maximum number of spikes
      in each timestep: queue=SpikeQueue(nsteps,maxevents)
    * At the beginning or end of each timestep: queue.next()
    * To get all spikes: events=queue.peek()
      It returns the indexes of all synapses receiving an event.
    * When a presynaptic spike is emitted:
      queue.insert(delay,offset,target)
      where delay is the array of synaptic delays of targets in timesteps,
      offset is the array of offsets within each timestep,
      target is the array of synapse indexes of targets.
      The offset is used to solve the problem of multiple synapses with the
      same delay. For example, if there are two target synapses 7 and 9 with delay
      2 timesteps: queue.insert([2,2],[0,1],[7,9])
    
    Thus, offsets are determined by delays. They could be either precalculated
    (faster), or determined at run time (saves memory). Note that if they
    are determined at run time, then it may be possible to also vectorize over
    presynaptic spikes.
    '''
    def __init__(self,nsteps,maxevents):
        # number of time steps, maximum number of spikes per time step
        self.X=zeros((nsteps,maxevents),dtype=int) # target synapses
        self.X_flat=self.X.reshape(nsteps*maxevents,)
        self.currenttime=0
        self.n=zeros(nsteps,dtype=int) # number of events in each time step
        
    def next(self):
        # Advance by one timestep
        self.n[self.currenttime]=0 # erase
        self.currenttime=(self.currenttime+1) % len(self.n)
        
    def peek(self):
        # Events in the current timestep
        return self.X[self.currenttime,:self.n[self.currenttime]]
    
    def offsets(self,delay):
        # Calculates offsets corresponding to a delay array
        # That's not a very efficient way to do it
        # (it's O(n*log(n)))
        # (not tested!)
        n_synapses_per_timestep=bincount(delays)
        ofs=hstack([arange(x) for x in n_synapses_per_timestep]) #very slow
        return ofs[argsort(delay)]
        # argsort is wrong (it's the converse) - multiply and add arange maybe?
        
    def insert(self,delay,offset,target):
        # Vectorized insertion of spike events
        # delay = delay in timestep
        # offset = offset within timestep
        # target = target synaptic index
        self.X_flat[(self.currenttime*self.X.shape[1]+offset+\
                     self.n[(self.currenttime+delay) % len(self.n)])\
                     % len(self.X)]=target
        # Update the size of the stacks
        self.n[(self.currenttime+delay) % len(self.n)]+=offset+1 # that's a trick

'''
The connection has arrays of synaptic variables (same as state matrix of
neuron groups). Two synaptic variables are the index of the postsynaptic neuron
and of the presynaptic neuron (i,j). (int32 or int16).

In addition, the connection must have, for each presynaptic neuron:
* list of target synapses (int32)
* corresponding delays in timesteps (int16)
* corresponding offsets (int16 is probably sufficient, or less)

These types (int32 etc) could be determined at construction time, or
at the time of conversion construction->connection (run time).

Same thing for postsynaptic neuron (for STDP)
This could also be determined at run time (depending on whether post=None or not)

Total memory:
* number of synapses * 12 * 2 (if bidirectional)
+ synaptic variables (weights)
'''

if __name__=='__main__':
    queue=SpikeQueue(5,30)
    delays=array([1,1,3,9,2,2,2,8,7])
    print argsort(delays)
    t1=time()
    for _ in range(10000):
        d=queue.offsets(delays)
    t2=time()
    print d
    print t2-t1