from base import *
from connection import *

__all__ = [
         'IdentityConnection',
         'MultiConnection',
         ]

class IdentityConnection(Connection):
    '''
    A :class:`Connection` between two groups of the same size, where neuron ``i`` in the
    source group is connected to neuron ``i`` in the target group.
    
    Initialised with arguments:
    
    ``source``, ``target``
        The source and target :class:`NeuronGroup` objects.
    ``state``
        The target state variable.
    ``weight``
        The weight of the synapse, must be a scalar.
    ``delay``
        Only homogeneous delays are allowed.
    
    The benefit of this class is that it has no storage requirements and is optimised for
    this special case.
    '''
    @check_units(delay=second)
    def __init__(self, source, target, state=0, weight=1, delay=0 * msecond):
        if (len(source) != len(target)):
            raise AttributeError, 'The connected (sub)groups must have the same size.'
        self.source = source # pointer to source group
        self.target = target # pointer to target group
        if type(state) == types.StringType: # named state variable
            self.nstate = target.get_var_index(state)
        else:
            self.nstate = state # target state index
        self.W = float(weight) # weight
        source.set_max_delay(delay)
        self.delay = int(delay / source.clock.dt) # Synaptic delay in time bins

    def propagate(self, spikes):
        '''
        Propagates the spikes to the target.
        '''
        self.target._S[self.nstate, spikes] += self.W

    def compress(self):
        pass


class MultiConnection(Connection):
    '''
    A hub for multiple connections with a common source group.
    '''
    def __init__(self, source, connections=[]):
        self.source = source
        self.connections = connections
        self.iscompressed = False
        self.delay = connections[0].delay

    def propagate(self, spikes):
        '''
        Propagates the spikes to the targets.
        '''
        for C in self.connections:
            C.propagate(spikes)

    def compress(self):
        if not self.iscompressed:
            for C in self.connections:
                C.compress()
            self.iscompressed = True
