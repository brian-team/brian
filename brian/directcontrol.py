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

__all__ = ['MultipleSpikeGeneratorGroup', 'SpikeGeneratorGroup', 'PulsePacket',
           'PoissonGroup', 'OfflinePoissonGroup', 'PoissonInput',
           'NewSpikeGeneratorGroup']

from neurongroup import *
from threshold import *
from stateupdater import *
from units import *
import random as pyrandom
from numpy import where, array, zeros, ones, inf, nonzero, tile, sum, isscalar, cumsum, hstack, bincount,  ceil, ndarray
from copy import copy
from clock import guess_clock
from utils.approximatecomparisons import *
import warnings
from operator import itemgetter
from log import *
import numpy
from numpy.random import exponential, randint, binomial
from connections import Connection

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
    def __init__(self, spiketimes, clock=None, period=None):
        """Pass spiketimes
        
        spiketimes is a list of lists, one list for each neuron
        in the group. Each sublist consists of the spike times.
        """
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
        for i in range(len(self.spiketimes)):
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
        for it, nextspiketime, i in zip(self.spiketimeiter, self.nextspiketime, range(len(self.spiketimes))):
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


class NewSpikeGeneratorGroup(NeuronGroup):
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
        allows you to use generator objects even though this is slower, see below). ``i`` can be an integer
        or an array (list of neurons that spike at the same time).
        If ``spiketimes`` is not a list or tuple, the pairs ``(i,t)`` need to be
        sorted in time. You can also pass a numpy array
        ``spiketimes`` where the first column of the array
        is the neuron indices, and the second column is the times in
        seconds. Alternatively you can pass a tuple with two arrays, the first one being the neuron indices and the second one times. WARNING: units are not checked in this case, the time array should be in seconds.
    ``clock``
        An optional clock to update with (omit to use the default clock).
    ``period``
        Optionally makes the spikes recur periodically with the given
        period. Note that iterator objects cannot be used as the ``spikelist``
        with a period as they cannot be reinitialised.
    ``gather=False``
        Set to True if you want to gather spike events that fall in the same
        timestep (makes the simulation faster if you have many events).
    ``sort=True``
        Set to False if your spike events are already sorted.
    
    Has an attribute:
    
    ``spiketimes``
        This can be used to reset the list of spike times, however the values of
        ``N``, ``clock`` and ``period`` cannot be changed. 
        
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
    of the neurons fires at an average rate of one every 5ms. Please note that as of 1.3.1, this behavior is preserved but will run slower than initializing with arrays, or lists.
    
    **Notes**
    
    Note that if a neuron fires more than one spike in a given interval ``dt``, additional
    spikes will be discarded. If you want them to stack, consider using the less efficient
    :class:`MultipleSpikeGeneratorGroup` object instead. A warning will be issued if this
    is detected.

    Also, if you want to use a SpikeGeneratorGroup with many spikes and/or neurons, please use an initialization with arrays
    
    Also note that if you pass a generator, then reinitialising the group will not have the
    expected effect because a generator object cannot be reinitialised. Instead, you should
    pass a callable object which returns a generator. In the example above, that would be
    done by calling::
    
        P = SpikeGeneratorGroup(10,nextspike)
        
    Whenever P is reinitialised, it will call ``nextspike()`` to create the required spike
    container.
    """
    def __init__(self, N, spiketimes, clock=None, period=None, 
                 sort=True, gather=None):
        clock = guess_clock(clock)
        self.period = period
        if gather:
            log_warn('brian.SpikeGeneratorGroup', 'SpikeGeneratorGroup\'s gather keyword use is deprecated')
        fallback = False # fall back on old SpikeGeneratorThreshold or not
        if isinstance(spiketimes, list):
            # spiketimes is a list of (i,t)
            idx, times = zip(*spiketimes)
            # the following try ... handles the case where spiketimes has index arrays
            # e.g spiketimes = [([0, 1], 0 * msecond), ([0, 1, 2], 2 * msecond)]
            # Notes:
            # - if there is always the same number of indices by array, its simple, it's just a matter of flattening
            # - if not, then it requires a for loop, and it's done in the except
            try:
                idx = array(idx, dtype = float)
                times = array(times, dtype = float)
                if idx.ndim > 1:
                    # simple case
                    times = tile(times.reshape((len(times), 1)), (idx.shape[1], 1)).flatten()
                    idx = idx.flatten()
            except ValueError:
                new_idx = []
                new_times = []
                for k, item in enumerate(idx):
                    if isinstance(item, list):
                        new_idx += item # append indices
                        new_times += [times[k]]*len(item)
                    else:
                        new_times += [times[k]]
                        new_idx += [item]
                idx = array(new_idx, dtype  = float)
                times = new_times
                times = array(times, dtype = float)
        elif isinstance(spiketimes, tuple):
            # spike times is a tuple with idx, times in arrays
            idx = spiketimes[0]
            times = spiketimes[1]
        elif isinstance(spiketimes, ndarray):
            # spiketimes is a ndarray, with first col is index and second time
            idx = spiketimes[:,0]
            times = spiketimes[:,1]
        else:
            log_warn('brian.SpikeGeneratorGroup', 'Using (slow) threshold because spiketimes is assumed to be a generator/iterator')
            # spiketimes is a callable object, so falling back on old SpikeGeneratorThreshold
            fallback = True

        if not fallback:
            thresh = FastSpikeGeneratorThreshold(N, idx, times, dt = clock.dt)
        else:
            thresh = SpikeGeneratorThreshold(N, spiketimes, period=period, sort=sort)
        
        NeuronGroup.__init__(self, N, model=LazyStateUpdater(), threshold=thresh, clock=clock)
 
    def reinit(self):
        super(NewSpikeGeneratorGroup, self).reinit()
        self._threshold.reinit()           
    
class FastSpikeGeneratorThreshold(Threshold):
    '''
    A faster version of the SpikeGeneratorThreshold where spikes are processed prior to the run (offline).
    
    '''
    ## Notes:
    #  - N is ignored (should it not?)
    def __init__(self, N, addr, timestamps, dt = None, period = None):
        self.set_offsets(addr, timestamps, dt = dt)
        self.period = period
        
    def set_offsets(self, I, T, dt = 1000):
        # Convert times into integers
        T = array(ceil(T/dt), dtype=int)
        # Put them into order
        # We use a field array to sort first by time and then by neuron index
        spikes = zeros(len(I), dtype=[('t', int), ('i', int)])
        spikes['t'] = T
        spikes['i'] = I
        spikes.sort(order=('t', 'i'))
        T = spikes['t']
        self.I = spikes['i']
        # Now for each timestep, we find the corresponding segment of I with
        # the spike indices for that timestep.
        # The idea of offsets is that the segment offsets[t]:offsets[t+1]
        # should give the spikes with time t, i.e. T[offsets[t]:offsets[t+1]]
        # should all be equal to t, and so then later we can return
        # I[offsets[t]:offsets[t+1]] at time t. It might take a bit of thinking
        # to see why this works. Since T is sorted, and bincount[i] returns the
        # number of elements of T equal to i, then j=cumsum(bincount(T))[t]
        # gives the first index in T where T[j]=t.
        self.offsets = hstack((0, cumsum(bincount(T))))
    
    def __call__(self, P):
        t = P.clock.t
        if self.period is not None:
            cp = int(t / self.period)
            if cp > self.curperiod:
                self.reinit()
                self.curperiod = cp
            t = t - cp * self.period
        dt = P.clock.dt
        t = int(round(t/dt))
        if t+1>=len(self.offsets):
            return array([], dtype=int)
        return self.I[self.offsets[t]:self.offsets[t+1]]
    
    def reinit(self):
        pass
    
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
        allows you to use generator objects, see below). ``i`` can be an integer
        or an array (list of neurons that spike at the same time).
        If ``spiketimes`` is not a list or tuple, the pairs ``(i,t)`` need to be
        sorted in time. You can also pass a numpy array
        ``spiketimes`` where the first column of the array
        is the neuron indices, and the second column is the times in
        seconds. WARNING: units are not checked in this case, and you need to
        ensure that the spikes are sorted.
    ``clock``
        An optional clock to update with (omit to use the default clock).
    ``period``
        Optionally makes the spikes recur periodically with the given
        period. Note that iterator objects cannot be used as the ``spikelist``
        with a period as they cannot be reinitialised.
    ``gather=False``
        Set to True if you want to gather spike events that fall in the same
        timestep (makes the simulation faster if you have many events).
    ``sort=True``
        Set to False if your spike events are already sorted.
    
    Has an attribute:
    
    ``spiketimes``
        This can be used to reset the list of spike times, however the values of
        ``N``, ``clock`` and ``period`` cannot be changed. 
        
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
    def __init__(self, N, spiketimes, clock=None, period=None, gather=False, sort=True):
        clock = guess_clock(clock)
        if gather: # assumes spike times are sorted
            spiketimes=self.gather(spiketimes,clock.dt)
            sort=False
        thresh = SpikeGeneratorThreshold(N, spiketimes, period=period, sort=sort)
        self.period = period
        NeuronGroup.__init__(self, N, model=LazyStateUpdater(), threshold=thresh, clock=clock)

    def gather(self,spiketimes,dt):
        '''
        Gathers spike events occurring in the same timestep.
        Assumes spikes are sorted (this could be ensured).
        
        Returns a new list of spike times (i,t), where i is an array of neurons
        (not just a single neuron) and t is a single floating value (time).
        
        Currently, this is rather slow. There is also a slight inconsistency
        with the non-gathered mechanism, where spikes are scheduled one timestep
        later. In addition, running a gathered group is slower! This is because
        of the bad design of the threshold.__call method.
        '''
        if isinstance(spiketimes, (list, tuple)): # First convert to (n,2) array
            spiketimes=array(spiketimes)
        times=array(spiketimes[:,1]/dt,dtype=int) # in units of dt
        neurons=array(spiketimes[:,0],dtype=int)
        u,indices=numpy.unique(times,return_index=True) # determine repeated time indices
        # Split over timesteps: list of (i,t) where i is an array of neurons
        new_spiketimes=[]
        for i in range(len(u)-1):
            new_spiketimes.append((neurons[indices[i]:indices[i+1]],float(u[i])*dt))
        new_spiketimes.append((neurons[indices[-1]:],float(u[-1])*dt))
        return new_spiketimes

    def reinit(self):
        super(SpikeGeneratorGroup, self).reinit()
        self._threshold.reinit()

    spiketimes = property(fget=lambda self:self._threshold.spiketimes,
                          fset=lambda self, value: self._threshold.set_spike_times(self._threshold.N, value, self._threshold.period))

    def __repr__(self):
        return "SpikeGeneratorGroup"


class SpikeGeneratorThreshold(Threshold):
    def __init__(self, N, spiketimes, period=None, sort=True):
        self.set_spike_times(N, spiketimes, period=period, sort=sort)

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

    def set_spike_times(self, N, spiketimes, period=None, sort=True):
        # N is the number of neurons, spiketimes is an iterable object of tuples (i,t) where
        # t is the spike time, and i is the neuron number. If spiketimes is a list or tuple,
        # then it will be sorted here.
        if isinstance(spiketimes, (list, tuple)) and sort:
            spiketimes = sorted(spiketimes, key=itemgetter(1))
        self.spiketimes = spiketimes
        self.N = N
        self.period = period
        self.reinit()
        
    def __call__(self, P):
        firing = zeros(self.N)
        t = P.clock.t
        if self.period is not None:
            cp = int(t / self.period)
            if cp > self.curperiod:
                self.reinit()
                self.curperiod = cp
            t = t - cp * self.period
        if isinstance(self.spiketimes, numpy.ndarray):
            t = float(t)
        while self.nextspiketime is not None and is_approx_less_than_or_equal(self.nextspiketime, t):
            if type(self.nextspikenumber)==int and firing[self.nextspikenumber]:
                log_warn('brian.SpikeGeneratorThreshold', 'Discarding multiple overlapping spikes')
            firing[self.nextspikenumber] = 1
            try:
                self.nextspikenumber, self.nextspiketime = self.spiketimeiter.next()
            except StopIteration:
                self.nextspiketime = None
        return where(firing)[0]

# The output of this function is fed into SpikeGeneratorGroup, consisting of
# time sorted pairs (t,i) where t is when neuron i fires
@check_units(t=second, n=1, sigma=second)
def PulsePacketGenerator(t, n, sigma):
    times = [pyrandom.gauss(t, sigma) for i in range(n)]
    times.sort()
    neuron = range(n)
    pyrandom.shuffle(neuron)
    return zip(neuron, times)


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
    @check_units(t=second, n=1, sigma=second)
    def __init__(self, t, n, sigma, clock=None):
        self.clock = guess_clock(clock)
        self.generate(t, n, sigma)

    def reinit(self):
        super(PulsePacket, self).reinit()
        self._threshold.reinit()
    @check_units(t=second, n=1, sigma=second)
    def generate(self, t, n, sigma):
        SpikeGeneratorGroup.__init__(self, n, PulsePacketGenerator(t, n, sigma), self.clock)

    def __repr__(self):
        return "Pulse packet neuron group"

class PoissonGroup(NeuronGroup):
    '''
    A group that generates independent Poisson spike trains.
    
    **Initialised as:** ::
    
        PoissonGroup(N,rates[,clock])
    
    with arguments:
    
    ``N``
        The number of neurons in the group
    ``rates``
        A scalar, array or function returning a scalar or array.
        The array should have the same length as the number of
        neurons in the group. The function should take one
        argument ``t`` the current simulation time.
    ``clock``
        The clock which the group will update with, do not
        specify to use the default clock.
    '''
    def __init__(self, N, rates=0 * hertz, clock=None):
        '''
        Initializes the group.
        P.rates gives the rates.
        '''
        NeuronGroup.__init__(self, N, model=LazyStateUpdater(), threshold=PoissonThreshold(),
                             clock=clock)
        if callable(rates): # a function is passed
            self._variable_rate = True
            self.rates = rates
            self._S0[0] = self.rates(self.clock.t)
        else:
            self._variable_rate = False
            self._S[0, :] = rates
            self._S0[0] = rates
        self.var_index = {'rate':0}

    def update(self):
        if self._variable_rate:
            self._S[0, :] = self.rates(self.clock.t)
        NeuronGroup.update(self)


class OfflinePoissonGroup(object): # This is weird, there is only an init method
    def __init__(self, N, rates, T):
        """
        Generates a Poisson group with N spike trains and given rates over the
        time window [0,T].
        """
        if isscalar(rates):
            rates = rates * ones(N)
        totalrate = sum(rates)
        isi = exponential(1 / totalrate, T * totalrate * 2)
        spikes = cumsum(isi)
        spikes = spikes[spikes <= T]
        neurons = randint(0, N, len(spikes))
        self.spiketimes = zip(neurons, spikes)


# Used in PoissonInput below
class EmptyGroup(object):
    def __init__(self, clock):
        self.clock = clock
    def get_spikes(self, delay):
        return None


class PoissonInput(Connection):
    _record = []
    
    def __init__(self, target, **kwargs):
        """
        Add a Poisson input to a NeuronGroup. Allows to efficiently simulate a large number of 
        independent Poisson inputs to a NeuronGroup variable, without simulating every synapse
        individually. The synaptic events are generated randomly during the simulation and
        are not preloaded and stored in memory (unless record=True is used).
        All the inputs must target the same variable, have the same frequency and same synaptic
        weight. You can use as many PoissonInput objects as you want, even targetting a same NeuronGroup.
        There is the possibility to consider time jitter in the presynaptic spikes, and
        synaptic unreliability. The inputs can also be recorded if needed. Finally, all
        neurons from the NeuronGroup receive independent realizations of Poisson spike trains,
        except if the keyword frozen=True is used, in which case all neurons receive the same
        Poisson input.
        
        Initialized with arguments:
        
        * target is the target NeuronGroup
        * n is the number of independent Poisson inputs
        * rate is the rate of each Poisson process
        * w is the synaptic weight
        * var is the name or the index of the synaptic variable of the NeuronGroup
        * jitter is None if there is no time jitter, otherwise it is (p, taujitter)
          where p is the number of synchronous spikes, and taujitter the jitter in second
        * reliability is None if the synapses are reliable, otherwise it is
          (p, alpha) where p is the number of synchronous spikes, and alpha is the
          reliability of the synapse (between 0 and 1)
        * record is True if the input has to be recorded
        * frozen is True if the input must be the same for all neurons of the NeuronGroup
        """
        self.source = EmptyGroup(target.clock)
        self.target = target
        self.N = len(self.target)
        self.clock = target.clock
        self.delay = None
        self.iscompressed = True
        self.delays = None # delay to wait for the j-th synchronous spike to occur after the last sync event, for target neuron i
        self.lastevent = -inf * ones(self.N) # time of the last event for target neuron i
        self.events = []
        self.recorded_events = []

        self.add_input(**kwargs)

    def add_input(self, n=None, rate=None, w=None, var=None,
                  jitter=None, reliability=None,
                  record=False,
                  frozen=False):
        """
        Add a Poisson input
        * n is the number of independent inputs
        * rate is the rate of each Poisson process
        * w is the synaptic weight
        * var is the name or the index of the synaptic variable of the NeuronGroup
        * jitter is None if there is no time jitter, otherwise it is (p, taujitter)
          where p is the number of synchronous spikes, and taujitter the jitter in second
        * reliability is None if the synapses are reliable, otherwise it is
          (p, alpha) where p is the number of synchronous spikes, and alpha is the
          reliability of the synapse (between 0 and 1)
        * record is True if this input has to be recorded
        * frozen is True if this input must be the same for all neurons of the NeuronGroup
          """

        self.n = n
        self.rate = rate
        self.w = w
        self.var = var
        self._jitter = jitter
        self.reliability = reliability
        self.record = record
        self.frozen = frozen

        if isinstance(var, str): # named state variable
            self.index = self.target.get_var_index(var)
        else:
            self.index = var

        #if jitter is not None:
        #    self.delays = zeros((jitter[0], self.N))

    @property
    def jitter(self):
        return self._jitter

    @jitter.setter
    def jitter(self, value):
        self._jitter = value
        if value is not None:
            self.delays = zeros((value[0], self.N))

    def propagate(self, spikes):
        i = 0
        
        n = self.n
        f = self.rate
        w = self.w
        var = self.var
        jitter = self.jitter
        reliability = self.reliability
        record = self.record
        frozen = self.frozen
        state = self.index
        
        if (jitter==None) and (reliability==None):
            if frozen:
                rnd = binomial(n=n, p=f * self.clock.dt)
                self.target._S[state, :] += w * rnd
                if rnd > 0:
                    self.events.append(self.clock.t)
            else:
                rnd = binomial(n=n, p=f * self.clock.dt, size=(self.N))
                self.target._S[state, :] += w * rnd
                ind = nonzero(rnd>0)[0]
                if record and len(ind)>0:
                    self.recorded_events.append((ind[0], self.clock.t))
        elif (jitter is not None):
            p = jitter[0]
            taujitter = jitter[1]
            if (p > 0) & (f > 0):
                taujitter = jitter[1]
                k = binomial(n=n, p=f * self.clock.dt, size=(self.N)) # number of synchronous events here, for every target neuron
                syncneurons = (k > 0) # neurons with a syncronous event here
                self.lastevent[syncneurons] = self.clock.t
                if taujitter == 0.0:
                    self.delays[:, syncneurons] = zeros((p, sum(syncneurons)))
                else:
                    self.delays[:, syncneurons] = exponential(scale=taujitter, size=(p, sum(syncneurons)))
                # Delayed spikes occur now
                lastevent = tile(self.lastevent, (p, 1))
                b = (abs(self.clock.t - (lastevent + self.delays)) <= (self.clock.dt / 2) * ones((p, self.N))) # delayed spikes occurring now
                weff = sum(b, axis=0) * w
                self.target._S[state, :] += weff
        elif (reliability is not None):
            p = reliability[0]
            alpha = reliability[1]
            if (p > 0) & (alpha > 0):
                weff = w * binomial(n=p, p=alpha)
                self.target._S[state, :] += weff * binomial(n=n, p=f * self.clock.dt, size=(self.N))

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
