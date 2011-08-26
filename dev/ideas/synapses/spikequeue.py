"""
Spike queues following BEP-21

TODO: Dynamic structure
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
    def __init__(self, nsteps, maxevents):
        # number of time steps, maximum number of spikes per time step
        self.X = zeros((nsteps, maxevents), dtype = int) # target synapses
        self.X_flat = self.X.reshape(nsteps*maxevents,)
        self.currenttime = 0
        self.n = zeros(nsteps, dtype = int) # number of events in each time step
        
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
        I = argsort(delay)
        xs = delay[I]
        J = xs[1:]!=xs[:-1]
        #K = xs[1:]==xs[:-1]
        A = hstack((0, cumsum(J)))
        #B = hstack((0, cumsum(K)))
        B = hstack((0, cumsum(-J)))
        BJ = hstack((0, B[J]))
        ei = B-BJ[A]
        ofs = zeros_like(delay)
        ofs[I] = ei
        return ofs
        
    def insert(self,delay,offset,target):
        # Vectorized insertion of spike events
        # delay = delay in timestep
        # offset = offset within timestep
        # target = target synaptic index
        timesteps=(self.currenttime+delay) % len(self.n)
        self.X_flat[(self.currenttime*self.X.shape[1]+offset+\
                     self.n[timesteps])\
                     % len(self.X)]=target
        # Update the size of the stacks
        self.n[timesteps]+=offset+1 # that's a trick
        # There should a re-sizing operation, if overflow
        
    def plot(self, display = True):
        for i in range(self.X.shape[0]):
            idx = (i + self.currenttime ) % self.X.shape[0]
            data = self.X[idx, :self.n[idx]]
            plot(idx * ones(len(data)), data, '.')
        if display:
            show()


    
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
    Nsynapses=4000*80 # in the CUBA example
    nspikes=160
    delays=randint(160,size=nspikes) # average number of spikes per dt in CUBA
    targets=randint(Nsynapses,size=nspikes)
    #print queue.offsets(delays)
    t1=time()
    for _ in range(10000): # 10000 timesteps per second
        d=queue.offsets(delays)
        queue.insert(delays,d,targets)
        queue.next()
        events=queue.peek()
        queue.plot()
    t2=time()
    print t2-t1
