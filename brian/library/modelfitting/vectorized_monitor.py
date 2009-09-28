from brian import *
from vectorized_neurongroup import *

class VectorizedSpikeMonitor(SpikeMonitor):
    def getvspikes(self):
    # Vectorized group: need to concatenate sliced trains
        if isinstance(self.source, VectorizedNeuronGroup):
            N = self.source.neuron_number
            overlap = self.source.overlap
            duration = self.source.duration
            vspikes = [(mod(i,N),(t-overlap)+i/N*(duration-overlap)*second) for i,t in self.spikes if t >= overlap]
            vspikes.sort(cmp=lambda x,y:2*int(x[1]>y[1])-1)
            return vspikes
    vspikes = property(fget=getvspikes)
    
    