from ..neurongroup import NeuronGroup
from ..threshold import Threshold
from ..stateupdater import LazyStateUpdater
import random as pyrandom
from numpy import where, array, zeros, ones, inf, nonzero, tile, sum, isscalar,\
                  cumsum, hstack, bincount,  ceil, ndarray, ascontiguousarray
from ..clock import guess_clock
from ..utils.approximatecomparisons import is_approx_less_than_or_equal
import warnings
from operator import itemgetter
from ..log import log_warn
import numpy
from numpy.random import exponential, randint, binomial
from ..connections import Connection
from itertools import izip

__all__ = ['MultipleSpikeGeneratorGroup']

class MultipleSpikeGeneratorGroup(NeuronGroup):
    """Emits spikes at given times
    
    .. warning::
        This function has been deprecated after Brian 1.3.1 and will be removed
        in a future release. Use :class:`SpikeGeneratorGroup` instead. To
        convert ``spiketimes`` for :class:`MultipleSpikeGeneratorGroup` into
        a form suitable for :class:`SpikeGeneratorGroup`, do::
        
            N = len(spiketimes)
            spiketimes = [(i, t) for i in xrange(N) for t in spiketimes[i]]
    
    **Initialised as:** ::
    
        MultipleSpikeGeneratorGroup(spiketimes[,clock[,period]])
    
    with arguments: 

    ``spiketimes``
        a list of spike time containers, one for each neuron in the group,
        although note that elements of ``spiketimes`` can also be callable objects which
        return spike time containers if you want to be able to reinitialise (see below).
        At it's simplest, ``spiketimes`` could be a list of lists, where ``spiketimes[0]`` contains
        the firing times for neuron 0, ``spiketimes[1]`` for neuron 1, etc. But, any iterable
        object can be passed, so ``spiketimes[0]`` could be a generator for example. Each
        spike time container should be sorted in time. If the containers are numpy arrays units
        will not be checked (times should be in seconds).
    ``clock``
        A clock, if omitted the default clock will be used.
    ``period``
        Optionally makes the spikes recur periodically with the given
        period. Note that iterator objects cannot be used as the ``spikelist``
        with a period as they cannot be reinitialised.
    
    Note that if two or more spike times fall within the same ``dt``, spikes will stack up
    and come out one per dt until the stack is exhausted. A warning will be generated
    if this happens.

    Also note that if you pass a generator, then reinitialising the group will not have the
    expected effect because a generator object cannot be reinitialised. Instead, you should
    pass a callable object which returns a generator, this will be called each time the
    object is reinitialised by calling the ``reinit()`` method.
        
    **Sample usage:** ::
    
        spiketimes = [[1*msecond, 2*msecond]]
        P = MultipleSpikeGeneratorGroup(spiketimes)
    """
    def __init__(self, spiketimes, clock=None, period=None):
        """Pass spiketimes
        
        spiketimes is a list of lists, one list for each neuron
        in the group. Each sublist consists of the spike times.
        """
        log_warn('brian', 'MultipleSpikeGeneratorGroup is deprecated, use SpikeGeneratorGroup instead')
        clock = guess_clock(clock)
        thresh = MultipleSpikeGeneratorThreshold(spiketimes, period=period)
        NeuronGroup.__init__(self, len(spiketimes), model=LazyStateUpdater(), threshold=thresh, clock=clock)

    def reinit(self):
        super(MultipleSpikeGeneratorGroup, self).reinit()
        self._threshold.reinit()

    def __repr__(self):
        return "MultipleSpikeGeneratorGroup"


class MultipleSpikeGeneratorThreshold(Threshold):
    def __init__(self, spiketimes, period=None):
        self.set_spike_times(spiketimes, period=period)

    def reinit(self):
        # spiketimes is a container where each element is an iterable container of spike times for
        # each neuron. We store the iterator for each neuron, and the next spike time if it exists
        # or None if there are no spikes or no more spikes
        def makeiter(obj):
            if callable(obj): return iter(obj())
            return iter(obj)
        self.spiketimeiter = [makeiter(st) for st in self.spiketimes]
        self.nextspiketime = [None for st in self.spiketimes]
        for i in xrange(len(self.spiketimes)):
            try:
                self.nextspiketime[i] = self.spiketimeiter[i].next()
            except StopIteration:
                pass
        self.curperiod = -1

    def set_spike_times(self, spiketimes, period=None):
        self.spiketimes = spiketimes
        self.period = period
        self.reinit()

    def __call__(self, P):
        firing = zeros(len(self.spiketimes))
        t = P.clock.t
        if self.period is not None:
            cp = int(t / self.period)
            if cp > self.curperiod:
                self.reinit()
                self.curperiod = cp
            t = t - cp * self.period
        # it is the iterator for neuron i, and nextspiketime is the stored time of the next spike
        for i, (it, nextspiketime) in enumerate(izip(self.spiketimeiter, self.nextspiketime)):
            # Note we use approximate equality testing because the clock t+=dt mechanism accumulates errors
            if isinstance(self.spiketimes[i], numpy.ndarray):
                curt = float(t)
            else:
                curt = t
            if nextspiketime is not None and is_approx_less_than_or_equal(nextspiketime, curt):
                firing[i] = 1
                try:
                    nextspiketime = it.next()
                    if is_approx_less_than_or_equal(nextspiketime, curt):
                        log_warn('brian.MultipleSpikeGeneratorThreshold', 'Stacking multiple firing times')
                except StopIteration:
                    nextspiketime = None
                self.nextspiketime[i] = nextspiketime
        return where(firing)[0]
