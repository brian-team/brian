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

INITIAL_MAXSPIKESPER_DT = 1 # I guess it could be larger no?
# This is a 2D circular array, but also a SpikeMonitor

__all__=['SpikeQueue']

class SpikeQueue(SpikeMonitor):
    '''
    * Initialization * 

    Initialized with a source NeuronGroup, a Synapses object (from which it fetches the delays), a maximum delay
    
    Arguments
    ``source'' NeuronGroup that is monitored
    ``synapses'' List of arrays of synapse indexes
    ``delays'' Array of delays corresponding to synapse indexes

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
    
    '''
    def __init__(self, source, synapses, delays,
                 max_delay = 0*ms, maxevents = INITIAL_MAXSPIKESPER_DT,
                 precompute_offsets = True):
        '''
        ``source'' is the neuron group that sends spikes
        ``synapses'' is a list of synapses (synapses[i]=array of synapse indexes for neuron i)
        ``delays'' is an array of delays (delays[k]=delay of synapse k)
        ``max_delay''
            The maximum delay (in second) of synaptic events. At run time, the
            structure is resized to the maximum delay in ``delays'', and thus
            the ``max_delay'' should only be specified if delays can change
            during the simulation (in which case offsets should not be
            precomputed).
        '''
        # SpikeMonitor structure
        self.source = source #NeuronGroup
        self.delays = delays
        self.synapses = synapses
        self._precompute_offsets=precompute_offsets
        
        # number of time steps, maximum number of spikes per time step
        nsteps = int(np.floor((max_delay)/(self.source.clock.dt)))+1
        self.X = zeros((nsteps, maxevents), dtype = self.synapses[0].dtype) # target synapses
        self.X_flat = self.X.reshape(nsteps*maxevents,)
        self.currenttime = 0
        self.n = zeros(nsteps, dtype = int) # number of events in each time step
        
        self._offsets = None # precalculated offsets
        
        super(SpikeQueue, self).__init__(source, 
                                         record = False)
        
        #useweave=get_global_preference('useweave')
        #compiler=get_global_preference('weavecompiler')

    def compress(self):
        '''
        Prepare the structure:
        * calculate maximum delay
        * calculate offsets
        '''
        if (self._offsets is None) and self._precompute_offsets:
            self.precompute_offsets()

        # Adjust the maximum delay and number of events per timestep if necessary
        nsteps=max(self.delays)+1
        maxevents=self.X.shape[1]
        if maxevents==INITIAL_MAXSPIKESPER_DT:
            maxevents=max(INITIAL_MAXSPIKESPER_DT,max([len(targets) for targets in self.synapses]))
        if (nsteps>self.X.shape[0]) or (maxevents>self.X.shape[1]):
            self.X = zeros((nsteps, maxevents), dtype = self.synapses[0].dtype) # target synapses
            self.X_flat = self.X.reshape(nsteps*maxevents,)
            self.n = zeros(nsteps, dtype = int) # number of events in each time step

    ################################ SPIKE QUEUE DATASTRUCTURE ######################
    def next(self):
        # Advance by one timestep
        self.n[self.currenttime]=0 # erase
        self.currenttime=(self.currenttime+1) % len(self.n)
        
    def peek(self):
        # Events in the current timestep      
        return self.X[self.currenttime,:self.n[self.currenttime]]
    
    def precompute_offsets(self):
        #t0 = time.time()
        self._offsets=[]
        for i in range(len(self.synapses)):
            delays=self.delays[self.synapses[i].data]
            self._offsets.append(self.offsets(delays))
        #log_debug('spikequeue.offsets', 'Offsets computed in '+str(time.time()-t0))
    
    def offsets(self, delay):
        '''
        Calculates offsets corresponding to a delay array
        '''
        # We use merge sort because it preserves the input order of equal
        # elements in the sorted output
        I = argsort(delay,kind='mergesort')
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
        
    def insert(self, delay, target, offset=None):
        # Vectorized insertion of spike events
        # delay = delay in timesteps
        # offset = offset within timestep
        # target = target synaptic index
        
        if offset is None:
            offset=self.offsets(delay)
        
        timesteps = (self.currenttime + delay) % len(self.n)
                
        # Compute new stack sizes:
        old_nevents = self.n[timesteps].copy() # because we need this for the final assignment, but we need to precompute the  new one to check for overflow
        self.n[timesteps] += offset+1 # that's a trick (to update stack size), plus we pre-compute it to check for overflow
        # Note: the trick can only work if offsets are ordered in the right way
        
        m = max(self.n[timesteps])+1 # If overflow, then at least one self.n is bigger than the size
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
        #log_debug('spikequeue', 'Resizing SpikeQueue')
        
    def propagate(self, spikes):
        if len(spikes):
            if self._offsets is None: # vectorise over synaptic events
                synaptic_events=hstack([self.synapses[i].data for i in spikes])
                if len(synaptic_events):
                    delay = self.delays[synaptic_events]
                    self.insert(delay, synaptic_events)
            else: # offsets are precomputed
                for i in spikes:
                    synaptic_events=self.synapses[i].data # assuming a dynamic array: could change at run time?    
                    if len(synaptic_events):
                        delay = self.delays[synaptic_events]
                        offsets = self._offsets[i]
                        self.insert(delay, synaptic_events, offsets)

    ######################################## UTILS    
    def plot(self, display = True):
        for i in range(self.X.shape[0]):
            idx = (i + self.currenttime ) % self.X.shape[0]
            data = self.X[idx, :self.n[idx]]
            plot(idx * ones(len(data)), data, '.')
        if display:
            show()

if __name__=='__main__':
    from synapses import *
    P=NeuronGroup(1,model='v:1')
    S=Synapses(P,model='w:1')
    queue=S.pre_queue
    #delays=array([4,2,2,1,6,2,5,9,6,9],dtype=int)
    s="9 6 6 5 1 7 8 2 6 0 9 6 8 3 6 6 1 1 2 6 6 8 6 4 4 1 4 9 4 7 1 3 4 4 8 4 7\
 1 3 0 4 4 2 5 7 2 5 6 0 6 8 5 7 1 7 0 9 2 1 9 5 9 4 3 5 7 2 5 8 8 7 9 9 8\
 8 9 1 5 8 3 7 8 4 3 7 4 7 6 2 5 5 3 8 6 1 2 7 5 9 7".split()
    delays=array([int(x) for x in s])
    offsets=queue.offsets(delays)
    n=zeros(max(delays)+1,dtype=int)
    print offsets
    n[delays]+=offsets+1
    print n