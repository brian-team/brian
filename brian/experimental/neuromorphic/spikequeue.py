"""
SpikeQueue

We need to find better names.
This is really a sketch.
"""
#from brian.utils.circular import SpikeContainer
#from brian.neurongroup import NeuronGroup
#from brian.directcontrol import SpikeGeneratorGroup
from brian import *

class SpikeQueue(SpikeContainer):
    '''
    Implements a vectorised queue of future spikes, as a SpikeContainer.
    
    What we want to do: push future spikes, without advancing the cursor.
    * We must keep track of the time of the current timestep (corresponding to
    cursor position), which would be used in get_spikes.
    * We would like to insert multiple timesteps at once, possibly using a
    list of timestamps and a list of neuron indices.
    * When we push spikes, sometimes the spikes should be in the same timestep
    as the last ones.
    
    This will then be used for a NeuronGroup, as a replacement of LS.
    Latency will then be set directly through the Connection object.
    '''
    def __init__(self, m, useweave=False, compiler=None):
        SpikeContainer.__init__(self,m, useweave, compiler)
        self._offset=0

    def push(self, spikes):
        SpikeContainer.push(self,spikes)
        self._offset-=1
    
    def advance(self):
        '''
        Advances by one timestep
        '''
        self._offset+=1
    
    def get_spikes(self, delay, origin, N):
        """
        Returns those spikes in self[delay] between origin and origin+N
        """
        return self.S.get_conditional(self.ind[-delay - 1] - self.S.cursor+self._offset, \
                                     self.ind[-delay] - self.S.cursor +self._offset+ self.S.n, \
                                     origin, origin + N, origin)

class SpikeQueueGroup(NeuronGroup):
    '''
    A group that sends spikes like SpikeGeneratorGroup, but in a vectorised
    way, forgetting past events.
    Initialised with a vector of spike times and a vector of corresponding
    neuron indices.
    '''
    def __init__(self, N, spiketimes, neurons, clock=None, period=None):
        clock = guess_clock(clock)
        self.period = period
        NeuronGroup.__init__(self, N, model=LazyStateUpdater(), clock=clock)

    def reinit(self):
        super(SpikeQueueGroup, self).reinit()
        self._threshold.reinit()

    def update(self):
        # We implement it here because we reimplement LS
        pass

    #spiketimes = property(fget=lambda self:self._threshold.spiketimes,
    #                      fset=lambda self, value: self._threshold.set_spike_times(self._threshold.N, value, self._threshold.period))
