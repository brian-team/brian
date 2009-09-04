# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
"""Direct controlling mechanisms

NeuronGroups and callable objects which allow direct
control over the behaviour of neurons.
"""

__all__ = ['MultipleSpikeGeneratorGroup','SpikeGeneratorGroup','PulsePacket']

from neurongroup import *
from threshold import *
from stateupdater import *
from units import *
import random as pyrandom
from numpy import where, array, zeros
from copy import copy
from clock import guess_clock
from utils.approximatecomparisons import *
import warnings
from operator import itemgetter
from log import *
import numpy

class MultipleSpikeGeneratorGroup(NeuronGroup):
    """Emits spikes at given times
    
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
    def __init__(self,spiketimes,clock=None,period=None):
        """Pass spiketimes
        
        spiketimes is a list of lists, one list for each neuron
        in the group. Each sublist consists of the spike times.
        """
        clock = guess_clock(clock)
        thresh = MultipleSpikeGeneratorThreshold(spiketimes,period=period)
        NeuronGroup.__init__(self,len(spiketimes),model=LazyStateUpdater(),threshold=thresh,clock=clock)
    def reinit(self):
        super(MultipleSpikeGeneratorGroup,self).reinit()
        self._threshold.reinit()        
    def __repr__(self):
        return "MultipleSpikeGeneratorGroup"

class MultipleSpikeGeneratorThreshold(Threshold):
    def __init__(self,spiketimes,period=None):
        self.set_spike_times(spiketimes,period=period)
    def reinit(self):
        # spiketimes is a container where each element is an iterable container of spike times for
        # each neuron. We store the iterator for each neuron, and the next spike time if it exists
        # or None if there are no spikes or no more spikes
        def makeiter(obj):
            if callable(obj): return iter(obj())
            return iter(obj)
        self.spiketimeiter = [makeiter(st) for st in self.spiketimes]
        self.nextspiketime = [None for st in self.spiketimes]
        for i in range(len(self.spiketimes)):
            try:
                self.nextspiketime[i] = self.spiketimeiter[i].next()
            except StopIteration:
                pass
        self.curperiod = -1
    def set_spike_times(self,spiketimes,period=None):
        self.spiketimes = spiketimes
        self.period = period
        self.reinit()
    def __call__(self,P):
        firing = zeros(len(self.spiketimes))
        t = P.clock.t
        if self.period is not None:
            cp = int(t/self.period)
            if cp>self.curperiod:
                self.reinit()
                self.curperiod=cp
            t = t - cp * self.period
        # it is the iterator for neuron i, and nextspiketime is the stored time of the next spike
        for it, nextspiketime, i in zip(self.spiketimeiter,self.nextspiketime,range(len(self.spiketimes))):
            # Note we use approximate equality testing because the clock t+=dt mechanism accumulates errors
            if isinstance(self.spiketimes[i],numpy.ndarray):
                curt = float(t)
            else:
                curt = t
            if nextspiketime is not None and is_approx_less_than_or_equal(nextspiketime,curt):
                firing[i]=1
                try:
                    nextspiketime = it.next()
                    if is_approx_less_than_or_equal(nextspiketime,curt):
                        log_warn('brian.MultipleSpikeGeneratorThreshold', 'Stacking multiple firing times')
                except StopIteration:
                    nextspiketime = None
                self.nextspiketime[i] = nextspiketime
        return where(firing)[0]

class SpikeGeneratorGroup(NeuronGroup):
    """Emits spikes at given times
    
    Initialised as::
    
        SpikeGeneratorGroup(N,spiketimes[,clock[,period]])
    
    with arguments:
    
    ``N``
        The number of neurons in the group.
    ``spiketimes``
        An object specifying which neurons should fire and when. It can be a container
        such as a ``list``, containing tuples ``(i,t)`` meaning neuron ``i`` fires at
        time ``t``, or a callable object which returns such a container (which
        allows you to use generator objects, see below). If ``spiketimes`` is not
        a list or tuple, the pairs ``(i,t)`` need to be sorted in time. You can
        also pass a numpy array ``spiketimes`` where the first column of the
        array is the neuron indices, and the second column is the times in
        seconds. WARNING: units are not checked in this case, and you need to
        ensure that the spikes are sorted.
    ``clock``
        An optional clock to update with (omit to use the default clock).
    ``period``
        Optionally makes the spikes recur periodically with the given
        period. Note that iterator objects cannot be used as the ``spikelist``
        with a period as they cannot be reinitialised.
    
    **Sample usages**
    
    The simplest usage would be a list of pairs ``(i,t)``::
    
        spiketimes = [(0,1*ms), (1,2*ms)]
        SpikeGeneratorGroup(N,spiketimes)
    
    A more complicated example would be to pass a generator::

        import random
        def nextspike():
            nexttime = random.uniform(0*ms,10*ms)
            while True:
                yield (random.randint(0,9),nexttime)
                nexttime = nexttime + random.uniform(0*ms,10*ms)
        P = SpikeGeneratorGroup(10,nextspike())
    
    This would give a neuron group ``P`` with 10 neurons, where a random one
    of the neurons fires at an average rate of one every 5ms.
    
    **Notes**
    
    Note that if a neuron fires more than one spike in a given interval ``dt``, additional
    spikes will be discarded. If you want them to stack, consider using the less efficient
    :class:`MultipleSpikeGeneratorGroup` object instead. A warning will be issued if this
    is detected.
    
    Also note that if you pass a generator, then reinitialising the group will not have the
    expected effect because a generator object cannot be reinitialised. Instead, you should
    pass a callable object which returns a generator. In the example above, that would be
    done by calling::
    
        P = SpikeGeneratorGroup(10,nextspike)
        
    Whenever P is reinitialised, it will call ``nextspike()`` to create the required spike
    container.
    """
    def __init__(self,N,spiketimes,clock=None,period=None):
        clock = guess_clock(clock)
        thresh = SpikeGeneratorThreshold(N,spiketimes,period=period)
        self.period = period
        NeuronGroup.__init__(self,N,model=LazyStateUpdater(),threshold=thresh,clock=clock)
        
    def reinit(self):
        super(SpikeGeneratorGroup,self).reinit()
        self._threshold.reinit()
        
    def __repr__(self):
        return "SpikeGeneratorGroup"

class SpikeGeneratorThreshold(Threshold):
    def __init__(self,N,spiketimes,period=None):
        self.set_spike_times(N,spiketimes,period=period)
        
    def reinit(self):
        def makeiter(obj):
            if callable(obj): return iter(obj())
            return iter(obj)
        self.spiketimeiter = makeiter(self.spiketimes)
        try:
            self.nextspikenumber, self.nextspiketime = self.spiketimeiter.next()
        except StopIteration:
            self.nextspiketime = None
            self.nextspikenumber = 0
        self.curperiod = -1
        
    def set_spike_times(self,N,spiketimes,period=None):
        # N is the number of neurons, spiketimes is an iterable object of tuples (i,t) where
        # t is the spike time, and i is the neuron number. If spiketimes is a list or tuple,
        # then it will be sorted here.
        if isinstance(spiketimes,(list,tuple)):
            spiketimes = sorted(spiketimes,key=itemgetter(1))
        self.spiketimes = spiketimes
        self.N = N
        self.period = period
        self.reinit()
        
    def __call__(self,P):
        firing = zeros(self.N)
        t = P.clock.t
        if self.period is not None:
            cp = int(t/self.period)
            if cp>self.curperiod:
                self.reinit()
                self.curperiod=cp
            t = t - cp * self.period
        if isinstance(self.spiketimes,numpy.ndarray):
            t = float(t)
        while self.nextspiketime is not None and is_approx_less_than_or_equal(self.nextspiketime,t):
            if firing[self.nextspikenumber]:
                log_warn('brian.SpikeGeneratorThreshold', 'Discarding multiple overlapping spikes')
            firing[self.nextspikenumber] = 1
            try:
                self.nextspikenumber, self.nextspiketime = self.spiketimeiter.next()
            except StopIteration:
                self.nextspiketime = None
        return where(firing)[0]

# The output of this function is fed into SpikeGeneratorGroup, consisting of
# time sorted pairs (t,i) where t is when neuron i fires
@check_units(t=second,n=1,sigma=second)
def PulsePacketGenerator(t,n,sigma):
    times = [pyrandom.gauss(t,sigma) for i in range(n)]
    times.sort()
    neuron = range(n)
    pyrandom.shuffle(neuron)
    return zip(neuron,times)

class PulsePacket(SpikeGeneratorGroup):
    """
    Fires a Gaussian distributed packet of n spikes with given spread
    
    **Initialised as:** :: 
    
        PulsePacket(t,n,sigma[,clock])
        
    with arguments:
    
    ``t``
        The mean firing time
    ``n``
        The number of spikes in the packet
    ``sigma``
        The standard deviation of the firing times.
    ``clock``
        The clock to use (omit to use default or local clock)
    
    **Methods**
    
    This class is derived from :class:`SpikeGeneratorGroup` and has all its
    methods as well as one additional method:
    
    .. method:: generate(t,n,sigma)
    
        Change the parameters and/or generate a new pulse packet.
    """
    @check_units(t=second,n=1,sigma=second)
    def __init__(self,t,n,sigma,clock=None):
        self.clock = guess_clock(clock)
        self.generate(t,n,sigma)
    def reinit(self):
        super(PulsePacket,self).reinit()
        self._threshold.reinit()
    @check_units(t=second,n=1,sigma=second)
    def generate(self,t,n,sigma):
        SpikeGeneratorGroup.__init__(self,n,PulsePacketGenerator(t,n,sigma),self.clock)
    def __repr__(self):
        return "Pulse packet neuron group"

def _test():
    import doctest
    doctest.testmod()

if __name__=="__main__":
    _test()
