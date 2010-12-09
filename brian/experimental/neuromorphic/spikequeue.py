"""
SpikeQueue
"""
from brian.utils.circular import SpikeContainer

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
