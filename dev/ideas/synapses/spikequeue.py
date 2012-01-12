"""
Spike queues following BEP-21.

Notes
-----
A SpikeQueue object will always be attached to a Synapses object. One important point is that delays and
synapse indexes can be set after the objects are created. Therefore, they cannot be passed at initialization time,
and this is why the Synapses object is passed.
To save one indirection, what should be stored is a view on the delay and synapse arrays.

The object is a SpikeMonitor on the source NeuronGroup. When it is called, spikes are fetched from the NeuronGroup
into the queue. The way it is currently done is highly inefficient, because it only uses the following mappings
    synapse -> delay
    synapse -> presynaptic i
but for efficiency, we need the mappings:
    presynaptic i -> synapse
    presynaptic i -> delay

We will still need the mapping: synapse -> presynaptic i
but not: synapse -> delay

Actually, does this need to be a Brian object? It could directly be called by
Synapses.

** There is no resizing for the maximum delay **
"""
from brian import * # remove this
from brian.stdunits import ms

INITIAL_MAXSPIKESPER_DT = 1
# This is a 2D circular array, but also a SpikeMonitor

class SpikeQueue(SpikeMonitor):
    '''
    * Initialization * 

    Initialized with a source NeuronGroup, a Synapses object (from which it fetches the delays), a maximum delay
    
    Arguments
    ``source'' self explanatory
    ``synapses'' self explanatory

    Keywords
    ``max_delay'' in seconds
    ``maxevents'' Maximum initial number of events in each timestep. Notice that the structure will grow dynamically of there are more events than that, so you shouldn't bother. 


    * Circular 2D array structure * 
    
    A spike queue is implemented as a circular 2D array.
    
    * At the beginning or end of each timestep: queue.next()
    * To get all spikes: events=queue.peek()
      It returns the indexes of all synapses receiving an event.
    * When a presynaptic spike is emitted, the following is executed:
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
    
    * SpikeMonitor structure * 
    
    It automatically updates the underlying structure by instantiating the propagate() method of the SpikeMonitor
    
    Ideas:
    ------
    * remove the max_delay keyword and have the structure created with another
      method (at run time)
    '''
    def __init__(self, source, synapses, 
                 max_delay = 0*ms, maxevents = INITIAL_MAXSPIKESPER_DT,
                 precompute_offsets = False):
        '''
        TODO:
        * precompute offsets
        * make it work for both pre/post
        * either source or synapses is not useful, no?
        '''
        # SpikeMonitor structure
        self.source = source #NeuronGroup
        self.synapses = synapses #Synapses
        
        self.max_delay = max_delay # do we need this?
        nsteps = int(np.floor((max_delay)/(self.source.clock.dt)))+1

        # number of time steps, maximum number of spikes per time step
        self.X = zeros((nsteps, maxevents), dtype = synapses.synapses_pre[0].dtype) # target synapses
        self.X_flat = self.X.reshape(nsteps*maxevents,)
        self.currenttime = 0
        self.n = zeros(nsteps, dtype = int) # number of events in each time step
        
        self._all_offsets = None
        self.precompute_offsets = precompute_offsets
        
        super(SpikeQueue, self).__init__(source, 
                                         record = False)

    ################################ SPIKE QUEUE DATASTRUCTURE ######################
    def next(self):
        # Advance by one timestep
        self.n[self.currenttime]=0 # erase
        self.currenttime=(self.currenttime+1) % len(self.n)
        
    def peek(self):
        # Events in the current timestep        
        return self.X[self.currenttime,:self.n[self.currenttime]]
    
    def compute_all_offsets(self, all_delays):
        #t0 = time.time()
        self._all_offsets = self.offsets(all_delays)
        #log_debug('spikequeue.offsets', 'Offsets computed in '+str(time.time()-t0))
    
    def offsets(self, delay):
        # Calculates offsets corresponding to a delay array
        # Maybe it could be precalculated? (an int16 array?)
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
        ofs[I] = array(ei,dtype=ofs.dtype) # maybe types should be signed?
        return ofs
        
    def insert(self, delay, offset, target):
        # Vectorized insertion of spike events
        # delay = delay in timestepp
        # offset = offset within timestep
        # target = target synaptic index
        
        timesteps = (self.currenttime + delay) % len(self.n)
        
        # Compute new stack sizes:
        old_nevents = self.n[timesteps].copy() # because we need this for the final assignment, but we need to precompute the  new one to check for overflow
        self.n[timesteps] += offset+1 # that's a trick (to update stack size), plus we pre-compute it to check for overflow
        
        m = max(self.n[timesteps]) # If overflow, then at least one self.n is bigger than the size
        if (m >= self.X.shape[1]):
            self.resize(m+1) # was m previously (not enough)
        
        self.X_flat[timesteps*self.X.shape[1]+offset+old_nevents]=target
        # Old code seemed wrong:
        #self.X_flat[(self.currenttime*self.X.shape[1]+offset+\
        #             old_nevents)\
        #             % len(self.X)]=target
        
    def resize(self, maxevents):
        '''
        Resizes the underlying data structure (number of columns = spikes per dt).
        max events will be rounded to the closest power of 2.
        '''
        
        # old and new sizes
        old_maxevents = self.X.shape[1]
        new_maxevents = 2**ceil(log2(maxevents)) # maybe 2 is too large
        # new array
        newX = zeros((self.X.shape[0], new_maxevents), dtype = self.X.dtype)
        newX[:, :old_maxevents] = self.X[:, :old_maxevents] # copy old data
        
        self.X = newX
        self.X_flat = self.X.reshape(self.X.shape[0]*new_maxevents,)
        
        log_debug('spikequeue', 'Resizing SpikeQueue')
        
    def propagate(self, spikes):
        '''
        TODO:
        * At this moment it only works in the forward direction. We need to have
        a specific variable instead of synapses_pre
        '''
        for i in spikes: # I don't see a way to avoid this loop at this moment
            synaptic_events=self.synapses.synapses_pre[i].data # assuming a dynamic array: could change at run time?    
            if len(synaptic_events):
                delay = self.synapses.delay_pre[synaptic_events] # but it could be post!
                if self.precompute_offsets:
                    #if self._all_offsets == None:
                    #    self.compute_all_offsets(self.synapses._statevector.delay) # change this
                    self.insert(delay, self._all_offsets[synaptic_events], synaptic_events)
                else:
                    offsets = self.offsets(delay)
                    self.insert(delay, offsets, synaptic_events)
            
    ######################################## UTILS    
    def plot(self, display = True):
        for i in range(self.X.shape[0]):
            idx = (i + self.currenttime ) % self.X.shape[0]
            data = self.X[idx, :self.n[idx]]
            plot(idx * ones(len(data)), data, '.')
        if display:
            show()

# We need to write some speed tests
if __name__=='__main__':
    pass