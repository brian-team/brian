from brian import *
import numpy

__all__ = ['RefractoryBeforeThresholdNeuronGroup', 'NeuronGroup']


class RefractoryBeforeThresholdNeuronGroup(NeuronGroup):
    '''
    NeuronGroup that performs refractoriness before thresholding
    
    If you have neurons that receiving very strong inputs enough to drive them
    from reset to threshold in a single time step dt then you can use this
    temporary hacked version of NeuronGroup to prevent spiking during the
    refractory period. Performance may not be great, and possibly the refractory
    period will be what you have given + dt. This should be fixed in a
    later release of Brian.
    '''
    def update(self):
        self._state_updater(self) # update the variables
        if isinstance(self._resetfun, Refractoriness):
            self._resetfun(self)
        spikes = self._threshold(self) # get spikes
        if not isinstance(spikes, numpy.ndarray):
            spikes = array(spikes, dtype=int)
        self.LS.push(spikes) # Store spikes

NeuronGroup = RefractoryBeforeThresholdNeuronGroup

if __name__ == '__main__':
    #from brian import NeuronGroup
    G = NeuronGroup(1, 'V:1', threshold=1, reset=0, refractory=5 * ms)
    @network_operation(when='before_groups')
    def f():
        G.V = 2
    M = StateMonitor(G, 'V', record=True, when='before_resets')
    sp = SpikeMonitor(G)
    run(40 * ms)
    print sp.spikes
    plot(M.times, M[0])
    show()
