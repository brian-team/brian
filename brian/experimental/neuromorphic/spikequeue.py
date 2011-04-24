"""
SpikeQueue

This is really a sketch.
"""
#from brian.utils.circular import SpikeContainer
#from brian.neurongroup import NeuronGroup
#from brian.directcontrol import SpikeGeneratorGroup
from brian import *
from numpy import unique
from time import time

__all__=['SpikeQueue']

class SpikeQueue(NeuronGroup):
    '''
    A group that sends spikes like SpikeGeneratorGroup, but in a vectorised
    way, forgetting past events.
    Initialised with a vector of spike times and a vector of corresponding
    neuron indices.
    '''
    def __init__(self, N, spiketimes, neurons, clock=None, check_sorted=True):
        clock = guess_clock(clock)
        NeuronGroup.__init__(self, N, model=LazyStateUpdater(), clock=clock)
        # Check if spike times are sorted
        if check_sorted: # Sorts the events if necessary
            if any(diff(spiketimes)<0): # not sorted
                ind=argsort(spiketimes)
                neurons,spiketimes=neurons[ind],spiketimes[ind]
        # Create the spike queue
        self.set_max_delay(max(spiketimes)) # This leaves space for the next spikes
        # Push the spikes
        self.LS.push(neurons) # This takes a bit of time (not sure why but this could be enhanced)
        # Set the cursors back
        self.LS.ind.advance(-1)
        self.LS.S.cursor=self.LS.ind[0]
        # Discretize spike times and make them relative to current time step
        spiketimes=array((spiketimes-clock.t)/clock.dt,dtype=int) # in units of dt
        # Calculate indices of spike groups
        u,indices=unique(spiketimes,return_index=True) # determine repeated time indices
        # Build vector of indices with -1 meaning: same index as previously
        x=-ones(max(u)+2) # maximum time index
        x[-1]=len(spiketimes) # last entry
        ## This is vectorized yet incredibly inefficient 
        #x[u]=indices
        #empty_ind=where(x<0)[0] # time bins with no spikes
        # This is really slow:
        #x[empty_ind]=indices[digitize(empty_ind,u)] # -1 are replaced with next positive entry
        
        # As a loop (This now takes about 30% of the whole construction time):
        # Perhaps it could be written in C
        x[u]=indices
        for i in where(x<0)[0][::-1]: # -1 are replaced with next positive entry
            x[i]=x[i+1]
        # x[0] is always 0; maybe this should be dropped
        self.LS.ind[0:len(x)]=x
        self.LS.ind[len(x)]=-1 # no more spike at that point
        self._stopped=False # True when no more spike

    def reinit(self):
        super(SpikeQueueGroup, self).reinit()
        
    def push_spike_times(self,spiketimes,neurons):
        '''
        Inserts future spike times
        For this we need to store the end position of future spikes
        '''
        pass

    def update(self):
        # LS.S contains the data (neurons that spike)
        # LS.ind is a circular vector with pointers to locations in LS.S,
        # one for each future time bin
        if (self.LS.ind[1]>=0) & (not self._stopped):
            ns=self.LS.ind[1]-self.LS.ind[0] # number of spikes in next bin
            self.LS.S.advance(ns)
        else:
            self._stopped=True
            self.LS.ind[1]=self.LS.ind[0]
        self.LS.ind.advance(1)
