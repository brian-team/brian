from ..connections.base import *
from ..connections.sparsematrix import *
from ..connections.connectionvector import *
from ..connections.constructionmatrix import *
from ..connections.connectionmatrix import *
from ..connections.construction import *
from ..connections.connection import *
from ..connections.delayconnection import *
from ..connections.propagation_c_code import *
import warnings
import numpy.random
network_operation = None # we import this when we need to because of order of import issues

__all__ = [
          'StochasticConnection',
          ]


class StochasticConnection(Connection):
    '''
    Connection which implements probabilistic (unreliable) synapses
    
    Initialised as for a :class:`Connection`, but with the additional
    keyword:
    
    ``reliability``
        Specifies the probability that a presynaptic spike triggers
        a postsynaptic current in the postsynaptic neuron.
    '''
    @check_units(delay=second)
    def __init__(self, source, target, state=0, delay=0 * msecond,
                 structure='dense',
                 reliability=1.0, **kwds):
        Connection.__init__(self, source, target, state=state, delay=delay,
                            structure=structure, **kwds)
        if (len(source) != len(target)):
            raise AttributeError, 'The connected (sub)groups must have the same size.'
        source.set_max_delay(delay)
        self.reliability = reliability

    def propagate(self, spikes):
        '''
        Propagates the spikes to the target.
        '''
        if len(spikes):
            rows = self.W[spikes, :] * (numpy.random.rand(len(spikes), self.W.shape[1]) <= self.reliability)
            for row in rows:
                self.target._S[self.nstate, :] += array(row, dtype=float32)

    def compress(self):
        pass
