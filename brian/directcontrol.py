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

__all__ = ['SpikeGeneratorGroup', 'PulsePacket',
           'PoissonGroup', 'OfflinePoissonGroup', 'PoissonInput']

from neurongroup import *
from threshold import *
from stateupdater import *
from units import *
import random as pyrandom
from numpy import where, array, zeros, ones, inf, nonzero, tile, sum, isscalar,\
                  cumsum, hstack, bincount,  ceil, ndarray, ascontiguousarray
from copy import copy
from clock import guess_clock
from utils.approximatecomparisons import *
import warnings
from operator import itemgetter
from log import *
import numpy
from numpy.random import exponential, randint, binomial
from connections import Connection
from itertools import izip


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
        timestep. (Deprecated since Brian 1.3.1)
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
    spikes will be discarded. A warning will be issued if this
    is detected.

    Also, if you want to use a SpikeGeneratorGroup with many spikes and/or neurons, please use an initialization with arrays.
    
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
        self.N = N
        self.period = period
        if gather:
            log_warn('brian.SpikeGeneratorGroup', 'SpikeGeneratorGroup\'s gather keyword use is deprecated')
        fallback = False # fall back on old SpikeGeneratorThreshold or not
        if isinstance(spiketimes, list):
            # spiketimes is a list of (i,t)
            if len(spiketimes):
                idx, times = zip(*spiketimes)
            else:
                idx, times = [], []
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
            thresh = FastSpikeGeneratorThreshold(N, idx, times, dt=clock.dt, period=period)
        else:
            thresh = SpikeGeneratorThreshold(N, spiketimes, period=period, sort=sort)
        
        if not hasattr(self, '_initialized'):
            NeuronGroup.__init__(self, N, model=LazyStateUpdater(), threshold=thresh, clock=clock)
            self._initialized = True
        else:
            self._threshold = thresh
 
    def reinit(self):
        super(SpikeGeneratorGroup, self).reinit()
        self._threshold.reinit()
        
    def get_spiketimes(self):
        return self._threshold.spiketimes
    
    def set_spiketimes(self, values):
        self.__init__(self.N, values, period = self.period)
    
    # changed due to the 2.5 issue
    spiketimes = property(get_spiketimes, set_spiketimes)



class FastSpikeGeneratorThreshold(Threshold):
    '''
    A faster version of the SpikeGeneratorThreshold where spikes are processed prior to the run (offline). It replaces the SpikeGeneratorThreshold as of 1.3.1.
    '''
    ## Notes:
    #  - N is ignored (should it not?)
    def __init__(self, N, addr, timestamps, dt = None, period=None):
        self.set_offsets(addr, timestamps, dt = dt)
        self.period = period
        self.dt = dt
        self.reinit()
        
    def set_offsets(self, I, T, dt = 1000):
        # Convert times into integers
        T = array(ceil(T/dt), dtype=int)
        # Put them into order
        # We use a field array to sort first by time and then by neuron index
        spikes = zeros(len(I), dtype=[('t', int), ('i', int)])
        spikes['t'] = T
        spikes['i'] = I
        spikes.sort(order=('t', 'i'))
        T = ascontiguousarray(spikes['t'])
        self.I = ascontiguousarray(spikes['i'])
        # Now for each timestep, we find the corresponding segment of I with
        # the spike indices for that timestep.
        # The idea of offsets is that the segment offsets[t]:offsets[t+1]
        # should give the spikes with time t, i.e. T[offsets[t]:offsets[t+1]]
        # should all be equal to t, and so then later we can return
        # I[offsets[t]:offsets[t+1]] at time t. It might take a bit of thinking
        # to see why this works. Since T is sorted, and bincount[i] returns the
        # number of elements of T equal to i, then j=cumsum(bincount(T))[t]
        # gives the first index in T where T[j]=t.
        if len(T):
            self.offsets = hstack((0, cumsum(bincount(T))))
        else:
            self.offsets = array([])
    
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
        self.curperiod = -1
        
    @property
    def spiketimes(self):
        # this is a pain to do! retrieve spike times from offsets
        res = []
        for k in range(len(self.offsets)-1):
            idx = self.I[self.offsets[k]:self.offsets[k+1]]
            ts = [k*self.dt]*len(idx)
            res += zip(idx, ts)
        return res

    def __repr__(self):
        return '<FastSpikeGeneratorThreshold>'

    def __str__(self):
        return 'Fast threshold mechanism for the SpikeGenerator group'
    

class SpikeGeneratorThreshold(Threshold):
    """
    Old threshold object for the SpikeGeneratorGroup
    
    **Notes**

    This version of the SpikeGeneratorThreshold object is deprecated, since version 1.3.1 of Brian it has been replaced in most cases by the FastSpikeGeneratorThreshold. 
    This is kept only as a fallback object for when a SpikeGeneratorGroup object is initialized with a generator or an iterator object (see the doc for SpikeGeneratorGroup for more details). Please note that since this implementation is slower, using a static data structure as an input to a SpikeGeneratorGroup is advised.
    """
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
    """
    Adds a Poisson input to a NeuronGroup. Allows to efficiently simulate a large number of 
    independent Poisson inputs to a NeuronGroup variable, without simulating every synapse
    individually. The synaptic events are generated randomly during the simulation and
    are not preloaded and stored in memory (unless record=True is used).
    All the inputs must target the same variable, have the same frequency and same synaptic
    weight. You can use as many PoissonInput objects as you want, even targetting a same NeuronGroup.
    There is the possibility to consider time jitter in the presynaptic spikes, and
    synaptic unreliability. The inputs can also be recorded if needed. Finally, all
    neurons from the NeuronGroup receive independent realizations of Poisson spike trains,
    except if the keyword freeze=True is used, in which case all neurons receive the same
    Poisson input.
    
    **Initialised as:** ::
    
        PoissonInput(target[, N][, rate][, weight][, state][, jitter][, reliability][, copies][, record][, freeze])
    
    with arguments:
    
    ``target``
        The target :class:`NeuronGroup`
    ``N``
        The number of independent Poisson inputs
    ``rate``
        The rate of each Poisson process
    ``weight``
        The synaptic weight
    ``state``
        The name or the index of the synaptic variable of the :class:`NeuronGroup`      
    ``jitter``
        is ``None`` by default. There is the possibility to consider ``copies`` presynaptic
        spikes at each Poisson event, randomly shifted according to an exponential law
        with parameter ``jitter=taujitter`` (in second).
    ``reliability`` 
        is ``None`` by default. There is the possibility to consider ``copies`` presynaptic
        spikes at each Poisson event, where each of these spikes is unreliable, i.e. it occurs
        with probability ``jitter=alpha`` (between 0 and 1).
    ``copies``
        The number of copies of each Poisson event. This is identical to ``weight=copies*w``, except
        if ``jitter`` or ``reliability`` are specified.
    ``record``
        ``True`` if the input has to be recorded. In this case, the recorded events are
        stored in the ``recorded_events`` attribute, as a list of pairs ``(i,t)`` where ``i`` is the
        neuron index and ``t`` is the event time.
    ``freeze``
        ``True`` if the input must be the same for all neurons of the :class:`NeuronGroup`
    """    
    _record = []
    
    def __init__(self, target, N=None, rate=None, weight=None, state=None,
                  jitter=None, reliability=None, copies=1,
                  record=False, freeze=False):
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

        self.n = N
        self.rate = rate
        self.w = weight
        self.var = state
        self._jitter = jitter
        
        if jitter is not None:
            self.delays = zeros((copies, self.N))
        
        self.reliability = reliability
        self.copies = copies
        self.record = record
        self.frozen = freeze
        
        if (jitter is not None) and (reliability is not None):
            raise Exception("Specifying both jitter and reliability is currently not supported.")

        if isinstance(state, str): # named state variable
            self.index = self.target.get_var_index(state)
        else:
            self.index = state

    def get_jitter(self):
        return self._jitter
    
    def set_jitter(self, value):
        self._jitter = value
        if value is not None:
            self.delays = zeros((self.copies, self.N))
    
    # changed due to the 2.5 issue
    jitter = property(get_jitter, set_jitter)
    

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
            p = self.copies
            taujitter = jitter
            if (p > 0) & (f > 0):
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
            p = self.copies
            alpha = reliability
            if (p > 0) & (alpha > 0):
                weff = w * binomial(n=p, p=alpha)
                self.target._S[state, :] += weff * binomial(n=n, p=f * self.clock.dt, size=(self.N))

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
