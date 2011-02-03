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
'''
Monitors (spikes and state variables).
* Tip: Spike monitors should have non significant impact on simulation time
if properly coded.
'''

__all__ = ['VanRossumMetric','SpikeMonitor', 'PopulationSpikeCounter', 'SpikeCounter', 'FileSpikeMonitor', 'StateMonitor', 'ISIHistogramMonitor', 'Monitor',
           'PopulationRateMonitor', 'StateSpikeMonitor', 'MultiStateMonitor', 'RecentStateMonitor', 'CoincidenceCounter', 'CoincidenceMatrixCounter', 'StateHistogramMonitor']

from units import *
from connections import Connection, SparseConnectionVector
from numpy import array, zeros, mean, histogram, linspace, tile, digitize,     \
        copy, ones, rint, exp, arange, convolve, argsort, mod, floor, asarray, \
        maximum, Inf, amin, amax, sort, nonzero, setdiff1d, diag, hstack, resize,\
         inf, var,tril,empty,float64,array,sum
from scipy.spatial.distance import sqeuclidean
from itertools import repeat, izip
from clock import guess_clock, EventClock, Clock
from network import NetworkOperation, network_operation
from stdunits import ms, Hz
from collections import defaultdict
import types
from operator import isSequenceType
from tools.statistics import firing_rate
from neurongroup import NeuronGroup
import bisect
from base import *
from time import time
try:
    import pylab, matplotlib
except:
    pass


from globalprefs import *
from scipy import weave


class Monitor(object):
    pass


class SpikeMonitor(Connection, Monitor):
    '''
    Counts or records spikes from a :class:`NeuronGroup`

    Initialised as one of::
    
        SpikeMonitor(source(,record=True))
        SpikeMonitor(source,function=function)
    
    Where:
    
    ``source``
        A :class:`NeuronGroup` to record from
    ``record``
        ``True`` or ``False`` to record all the spikes or just summary
        statistics.
    ``function``
        A function ``f(spikes)`` which is passed the array of neuron
        numbers that have fired called each step, to define
        custom spike monitoring.
    
    Has three attributes:
    
    ``nspikes``
        The number of recorded spikes
    ``spikes``
        A time ordered list of pairs ``(i,t)`` where neuron ``i`` fired
        at time ``t``.
    ``spiketimes``
        A dictionary with keys the indices of the neurons, and values an
        array of the spike times of that neuron. For example,
        ``t=M.spiketimes[3]`` gives the spike times for neuron 3.

    For ``M`` a :class:`SpikeMonitor`, you can also write:
    
    ``M[i]``
        An array of the spike times of neuron ``i``.

    Notes:

    :class:`SpikeMonitor` is subclassed from :class:`Connection`.
    To define a custom monitor, either define a subclass and
    rewrite the ``propagate`` method, or pass the monitoring function
    as an argument (``function=myfunction``, with ``def myfunction(spikes):...``)
    '''
    # isn't there a units problem here for delay?
    def __init__(self, source, record=True, delay=0, function=None):
        # recordspikes > record?
        self.source = source # pointer to source group
        self.target = None
        self.nspikes = 0
        self.spikes = []
        self.record = record
        self.W = None # should we just remove this variable?
        source.set_max_delay(delay)
        self.delay = int(delay / source.clock.dt) # Synaptic delay in time bins
        self._newspikes = True
        if function != None:
            self.propagate = function

    def reinit(self):
        """
        Clears all monitored spikes
        """
        self.nspikes = 0
        self.spikes = []

    def propagate(self, spikes):
        '''
        Deals with the spikes.
        Overload this function to store or process spikes.
        Default: counts the spikes (variable nspikes)
        '''
        if len(spikes):
            self._newspikes = True
        self.nspikes += len(spikes)
        if self.record:
            self.spikes += zip(spikes, repeat(self.source.clock.t))

    def origin(self, P, Q):

        '''
        Returns the starting coordinate of the given groups in
        the connection matrix W.
        '''
        return (P.origin - self.source.origin, 0)

    def compress(self):
        pass

    def __getitem__(self, i):
        return array([t for j, t in self.spikes if j == i])

    def getspiketimes(self):
        if self._newspikes:
            self._newspikes = False
            self._spiketimes = {}
            for i in xrange(len(self.source)):
                self._spiketimes[i] = []
            for i, t in self.spikes:
                self._spiketimes[i].append(float(t))
            for i in xrange(len(self.source)):
                self._spiketimes[i] = array(self._spiketimes[i])
        return self._spiketimes
    spiketimes = property(fget=getspiketimes)

#    def getvspikes(self):
#        if isinstance(self.source, VectorizedNeuronGroup):
#            N = self.source.neuron_number
#            overlap = self.source.overlap
#            duration = self.source.duration
#            vspikes = [(mod(i,N),(t-overlap)+i/N*(duration-overlap)*second) for (i,t) in self.spikes if t >= overlap]
#            vspikes.sort(cmp=lambda x,y:2*int(x[1]>y[1])-1)
#            return vspikes
#    concatenated_spikes = property(fget=getvspikes)


class AutoCorrelogram(SpikeMonitor):
    '''
    Calculates autocorrelograms for the selected neurons (online).
    
    Initialised as::
    
        AutoCorrelogram(source,record=[1,2,3], delay=10*ms)
    
    where ``delay`` is the size of the autocorrelogram.
    
    NOT FINISHED 
    '''
    def __init__(self, source, record=True, delay=0):
        SpikeMonitor.__init__(self, source, record=record, delay=delay)
        self.reinit()
        if record is not False:
            if record is not True and not isinstance(record, int):
                self.recordindex = dict((i, j) for i, j in zip(self.record, range(len(self.record))))

    def reinit(self):
        if self.record == True:
            self._autocorrelogram = zeros((len(self.record), len(self.source)))
        else:
            self._autocorrelogram = zeros((len(self.record), self.delay))

    def propagate(self, spikes):
        spikes_set = set(spikes)
        if self.record == True:
            for i in xrange(self.delay): # Not a brilliant implementation
                self._autocorrelogram[spikes_set.intersection(self.source.LS[i]), i] += 1

    def __getitem__(self, i):
        # TODO: returns the autocorrelogram of neuron i
        pass


class PopulationSpikeCounter(SpikeMonitor):
    '''
    Counts spikes from a :class:`NeuronGroup`

    Initialised as::
    
        PopulationSpikeCounter(source)
    
    With argument:
    
    ``source``
        A :class:`NeuronGroup` to record from
    
    Has one attribute:
    
    ``nspikes``
        The number of recorded spikes
    '''
    def __init__(self, source, delay=0):
        SpikeMonitor.__init__(self, source, record=False, delay=delay)


class SpikeCounter(PopulationSpikeCounter):
    '''
    Counts spikes from a :class:`NeuronGroup`

    Initialised as::
    
        SpikeCounter(source)
    
    With argument:
    
    ``source``
        A :class:`NeuronGroup` to record from
    
    Has two attributes:
    
    ``nspikes``
        The number of recorded spikes
    ``count``
        An array of spike counts for each neuron
    
    For a :class:`SpikeCounter` ``M`` you can also write ``M[i]`` for the
    number of spikes counted for neuron ``i``.
    '''
    def __init__(self, source):
        PopulationSpikeCounter.__init__(self, source)
        self.count = zeros(len(source), dtype=int)

    def __getitem__(self, i):
        return int(self.count[i])

    def propagate(self, spikes):
        PopulationSpikeCounter.propagate(self, spikes)
        self.count[spikes] += 1

    def reinit(self):
        self.count[:] = 0
        PopulationSpikeCounter.reinit(self)


class StateSpikeMonitor(SpikeMonitor):
    '''
    Counts or records spikes and state variables at spike times from a :class:`NeuronGroup`

    Initialised as::
    
        StateSpikeMonitor(source, var)
    
    Where:
    
    ``source``
        A :class:`NeuronGroup` to record from
    ``var``
        The variable name or number to record from, or a tuple of variable names or numbers
        if you want to record multiple variables for each spike.
    
    Has two attributes:
    
    .. attribute:: nspikes
    
        The number of recorded spikes
        
    .. attribute:: spikes
    
        A time ordered list of tuples ``(i,t,v)`` where neuron ``i`` fired
        at time ``t`` and the specified variable had value ``v``. If you
        specify multiple variables, each tuple will be of the form
        ``(i,t,v0,v1,v2,...)`` where the ``vi`` are the values corresponding
        in order to the variables you specified in the ``var`` keyword.
    
    And two methods:
    
    .. method:: times(i=None)
    
        Returns an array of the spike times for the whole monitored
        group, or just for neuron ``i`` if specified.
    
    .. method:: values(var, i=None)
    
        Returns an array of the values of variable ``var`` for the
        whole monitored group, or just for neuron ``i`` if specified.
    '''
    def __init__(self, source, var):
        SpikeMonitor.__init__(self, source)
        if isinstance(var, (str, int)) or not isSequenceType(var):
            var = (var,)
        self._varnames = var
        self._vars = [source.state_(v) for v in var]
        self._varindex = dict((v, i + 2) for i, v in enumerate(var))
        self._units = [source.unit(v) for v in var]

    def propagate(self, spikes):
        self.nspikes += len(spikes)
        recordedstate = [ [x * u for x in v[spikes]] for v, u in izip(self._vars, self._units) ]
        self.spikes += zip(spikes, repeat(self.source.clock.t), *recordedstate)

    def __getitem__(self, i):
        return NotImplemented # don't use the version from SpikeMonitor

    def times(self, i=None):
        '''Returns the spike times (of neuron ``i`` if specified)'''
        if i is not None:
            return array([x[1] for x in self.spikes if x[0] == i])
        else:
            return array([x[1] for x in self.spikes])

    def values(self, var, i=None):
        '''Returns the recorded values of ``var`` (for spikes from neuron ``i`` if specified)'''
        v = self._varindex[var]
        if i is not None:
            return array([x[v] for x in self.spikes if x[0] == i])
        else:
            return array([x[v] for x in self.spikes])


class HistogramMonitorBase(SpikeMonitor):
    pass


class ISIHistogramMonitor(HistogramMonitorBase):
    '''
    Records the interspike interval histograms of a group.
    
    Initialised as::
    
        ISIHistogramMonitor(source, bins)
    
    ``source``
        The source group to record from.
    ``bins``
        The lower bounds for each bin, so that e.g.
        ``bins = [0*ms, 10*ms, 20*ms]`` would correspond to
        bins with intervals 0-10ms, 10-20ms and
        20+ms.
        
    Has properties:
    
    ``bins``
        The ``bins`` array passed at initialisation.
    ``count``
        An array of length ``len(bins)`` counting how many ISIs
        were in each bin.
    
    This object can be passed directly to the plotting function
    :func:`hist_plot`.
    '''
    def __init__(self, source, bins, delay=0):
        SpikeMonitor.__init__(self, source, delay)
        self.bins = array(bins)
        self.reinit()

    def reinit(self):
        super(ISIHistogramMonitor, self).reinit()
        self.count = zeros(len(self.bins))
        self.LS = 1000 * second * ones(len(self.source))

    def propagate(self, spikes):
        super(ISIHistogramMonitor, self).propagate(spikes)
        isi = self.source.clock.t - self.LS[spikes]
        self.LS[spikes] = self.source.clock.t
        # all this nonsense is necessary to deal with the fact that
        # numpy changed the semantics of histogram in 1.2.0 or thereabouts
        try:
            h, a = histogram(isi, self.bins, new=True)
        except TypeError:
            h, a = histogram(isi, self.bins)
        if len(h) == len(self.count):
            self.count += h
        else:
            self.count[:-1] += h
            self.count[-1] += len(isi) - sum(h)


class FileSpikeMonitor(SpikeMonitor):
    """Records spikes to a file

    Initialised as::
    
        FileSpikeMonitor(source, filename[, record=False])
    
    Does everything that a :class:`SpikeMonitor` does except also records
    the spikes to the named file. note that spikes are recorded
    as an ASCII file of lines each of the form:
    
        ``i, t``
    
    Where ``i`` is the neuron that fired, and ``t`` is the time in seconds.
    
    Has one additional method:
    
    ``close_file()``
        Closes the file manually (will happen automatically when
        the program ends).
    """
    def __init__(self, source, filename, record=False, delay=0):
        super(FileSpikeMonitor, self).__init__(source, record, delay)
        self.filename = filename
        self.f = open(filename, 'w')

    def reinit(self):
        self.close_file()
        self.f = open(self.filename, 'w')

    def propagate(self, spikes):
        super(FileSpikeMonitor, self).propagate(spikes)
        for i in spikes:
            self.f.write(str(i) + ", " + str(float(self.source.clock.t)) + "\n")

    def close_file(self):
        self.f.close()


class PopulationRateMonitor(SpikeMonitor):
    '''
    Monitors and stores the (time-varying) population rate
    
    Initialised as::
    
        PopulationRateMonitor(source,bin)
    
    Records the average activity of the group for every bin.
    
    Properties:
    
    ``rate``, ``rate_``
        An array of the rates in Hz.    
    ``times``, ``times_``
        The times of the bins.
    ``bin``
        The duration of a bin (in second).
    '''
    times = property(fget=lambda self:array(self._times))
    times_ = times
    rate = property(fget=lambda self:array(self._rate))
    rate_ = rate

    def __init__(self, source, bin=None):
        SpikeMonitor.__init__(self, source)
        if bin:
            self._bin = int(bin / source.clock.dt)
        else:
            self._bin = 1 # bin size in number
        self._rate = []
        self._times = []
        self._curstep = 0
        self._clock = source.clock
        self._factor = 1. / float(self._bin * source.clock.dt * len(source))

    def reinit(self):
        SpikeMonitor.reinit(self)
        self._rate = []
        self._times = []
        self._curstep = 0

    def propagate(self, spikes):
        if self._curstep == 0:
            self._rate.append(0.)
            self._times.append(self._clock._t) # +.5*bin?
            self._curstep = self._bin
        self._rate[-1] += len(spikes) * self._factor
        self._curstep -= 1

    def smooth_rate(self, width=None, filter='gaussian'):
        """
        Returns a smoothed version of the vector of rates,
        convolving the rates with a filter (gaussian or flat)
        with the given width.
        """
        if width is None: # automatic with Shinomoto's algorithms
            if filter=='flat':
                """ (Experimental)
                If width is not given and the filter is flat, then the bin
                size is automatically chosen using Shimazaki and Shinomoto's method:
                  Shimazaki and Shinomoto, A method for selecting the bin size of a time histogram
                  Neural Computation 19(6), 1503-1527, 2007
                  http://dx.doi.org/10.1162/neco.2007.19.6.1503
                """
                # Shinomoto's method to find the optimal bin size. Adapted from:
                # Shimazaki and Shinomoto, A method for selecting the bin size of a time histogram
                # Neural Computation 19(6), 1503-1527, 2007
                # http://dx.doi.org/10.1162/neco.2007.19.6.1503
                counts=array(self._rate)/self._factor
                best_value=inf
                for nbins in range(2,500): # possible number of bins (maybe a less brutal optimization?)
                    binsize=len(counts)/nbins
                    x=resize(counts,(len(counts)/binsize,binsize))
                    #x.reshape((x.size,1))[len(counts):]=0 # unnecessary because smaller
                    x=x.sum(1) # x is the histogram with nbins bins
                    K=mean(x) # average number of spikes per recording bin
                    value=(2*K-var(x))/binsize**2
                    if value<best_value:
                        best_value=value
                        width_dt=binsize
                        nb=nbins
                #print width_dt,nb
            else:
                raise AttributeError,"Automatic width selection is not implemented yet!"
        else:
            width_dt = int(width / (self._bin * self._clock.dt)) # width in number of bins
        #print width_dt
        window = {'gaussian': exp(-arange(-2 * width_dt, 2 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2)),
                'flat': ones(width_dt)}[filter]
        return convolve(self.rate_, window * 1. / sum(window), mode='same')


class StateMonitor(NetworkOperation, Monitor):
    '''
    Records the values of a state variable from a :class:`NeuronGroup`.

    Initialise as::
    
        StateMonitor(P,varname(,record=False)
            (,when='end)(,timestep=1)(,clock=clock))
    
    Where:
    
    ``P``
        The group to be recorded from
    ``varname``
        The state variable name or number to be recorded
    ``record``
        What to record. The default value is ``False`` and the monitor will
        only record summary statistics for the variable. You can choose
        ``record=integer`` to record every value of the neuron with that
        number, ``record=``list of integers to record every value of each of
        those neurons, or ``record=True`` to record every value of every
        neuron (although beware that this may use a lot of memory).
    ``when``
        When the recording should be made in the network update, possible
        values are any of the strings: ``'start'``, ``'before_groups'``, ``'after_groups'``,
        ``'before_connections'``, ``'after_connections'``, ``'before_resets'``,
        ``'after_resets'``, ``'end'`` (in order of when they are run).
    ``timestep``
        A recording will be made each timestep clock updates (so ``timestep``
        should be an integer).
    ``clock``
        A clock for the update schedule, use this if you have specified a
        clock other than the default one in your network, or to update at a
        lower frequency than the update cycle. Note though that if the clock
        here is different from the main clock, the when parameter will not
        be taken into account, as network updates are done clock by clock.
        Use the ``timestep`` parameter if you need recordings to be made at a
        precise point in the network update step.

    The :class:`StateMonitor` object has the following properties:

    ``times``
        The times at which recordings were made
    ``mean``
        The mean value of the state variable for every neuron in the
        group (not just the ones specified in the ``record`` keyword)
    ``var``
        The unbiased estimate of the variances, as in ``mean``
    ``std``
        The square root of ``var``, as in ``mean``
    ``values``
        A 2D array of the values of all the recorded neurons, each row is a
        single neuron's values.
        
    In addition, if :class:`M`` is a :class:`StateMonitor` object, you write::
    
        M[i]
    
    for the recorded values of neuron ``i`` (if it was specified with the
    ``record`` keyword). It returns a numpy array.
    
    Methods:
    
    .. method:: plot([indices=None[, cmap=None[, refresh=None[, showlast=None[, redraw=True]]]]])
        
        Plots the recorded values using pylab. You can specify an index or
        list of indices, otherwise all the recorded values will be plotted.
        The graph plotted will have legends of the form ``name[i]`` for
        ``name`` the variable name, and ``i`` the neuron index. If cmap is
        specified then the colours will be set according to the matplotlib
        colormap cmap. ``refresh`` specifies how often (in simulation time)
        you would like the plot to refresh. Note that this will only work if
        pylab is in interactive mode, to ensure this call the pylab ``ion()``
        command. If you are using the ``refresh`` option, ``showlast`` specifies
        a fixed time window to display (e.g. the last 100ms).
        If you are using more than one realtime monitor, only one of them needs
        to issue a redraw command, therefore set ``redraw=False`` for all but
        one of them.
        
        Note that with some IDEs, interactive plotting will not work with the
        default matplotlib backend, try doing something like this at the
        beginning of your script (before importing brian)::
        
            import matplotlib
            matplotlib.use('WXAgg')
            
        You may need to experiment, try WXAgg, GTKAgg, QTAgg, TkAgg.
        
    .. method:: insert_spikes(spikemonitor[, value=0])

        Inserts spikes into recorded traces (for plotting). State values
        at spike times are replaced with the given value (peak value of spike).
    '''
    mean = property(fget=lambda self:self._mu / self.N)
    _mean = mean
    mean_ = _mean
    var = property(fget=lambda self:(self._sqr - self.N * self.mean_ ** 2) / (self.N - 1))
    var_ = var
    std = property(fget=lambda self:self.var ** .5)
    std_ = std
    times = property(fget=lambda self:array(self._times))
    times_ = times
    values = property(fget=lambda self:self.getvalues())
    values_ = values

    def __init__(self, P, varname, clock=None, record=False, timestep=1, when='end'):
        '''
        -- P is the neuron group
        -- varname is the variable name
        -- record can be one of:
           - an integer, in which case the value of the state of the corresponding
           neuron will be recorded in the list self._values
           - an array or list of integers, in which case the value of the states
           of the corresponding neurons will be recorded and can be individually
           accessed by calling self[i] where i is the neuron number
           - True, in which case the state of all neurons is recorded, and can be
           individually accessed by calling self[i]
           - False, in which case only the mean and variance are recorded (.mean, .var, .std)
        -- timestep defines how often a recording is made (e.g. if you have a very
           small dt, you might not want to record every value of the variable), it
           is an integer (multiple of the clock dt)
        '''
        NetworkOperation.__init__(self, None, clock=clock, when=when)
        self.record = record
        self.clock = guess_clock(clock)
        if record is not False:
            if record is not True and not isinstance(record, int):
                self.recordindex = dict((i, j) for i, j in zip(self.record, range(len(self.record))))
        self.timestep = timestep
        self.curtimestep = timestep
        self._values = None
        self.P = P
        self.varname = varname
        self.N = 0 # number of steps
        self._recordstep = 0
        if record is False:
            self._mu = zeros(len(P)) # sum
            self._sqr = zeros(len(P)) # sum of squares
        self.unit = 1.0 * P.unit(varname)
        self.reinit()

    def __call__(self):
        '''
        This function is called every time step.
        '''
        V = self.P.state_(self.varname)
        if self.record is False:
            self._mu += V
            self._sqr += V * V
        elif self.curtimestep == self.timestep:
            i = self._recordstep
            if not isinstance(self.record, bool):
                self._values.append(V[self.record])
            elif self.record is True:
                self._values.append(V.copy())
            self._times.append(self.clock._t)
            self._recordstep += 1
        self.curtimestep -= 1
        if self.curtimestep == 0: self.curtimestep = self.timestep
        self.N += 1

    def __getitem__(self, i):
        """Returns the recorded values of the state of neuron i as an array
        """
        if self.record is False:
            raise IndexError('Neuron ' + str(i) + ' was not recorded.')
        if self.record is not True:
            if isinstance(self.record, int) and self.record != i or (not isinstance(self.record, int) and i not in self.record):
                raise IndexError('Neuron ' + str(i) + ' was not recorded.')
            try:
                return self.values[self.recordindex[i]]
            except:
                if i == self.record:
                    return self.values[0]
                else:
                    raise
        elif self.record is True:
            return self.values[i]

    def getvalues(self):
        if len(self._values):
            newvalues = array(self._values)
            if len(newvalues.shape)==1:
                newvalues.shape = (1, newvalues.size)
            else:
                newvalues = newvalues.T
            values = hstack((self._values_cache, newvalues))
            self._values_cache = values
            self._values = []
        else:
            values = self._values_cache    
        return values
    getvalues_ = getvalues

    def reinit(self):
        self._values = []
        self._times = []
        ri = self.get_record_indices()
        self._values_cache = zeros((len(ri), 0))
        self.N = 0
        self._recordstep = 0
        self._mu = zeros(len(self.P))
        self._sqr = zeros(len(self.P))

    def get_record_indices(self):
        """Returns the list of neuron numbers which were recorded.
        """
        if self.record is False:
            return []
        elif self.record is True:
            return arange(len(self.P))
        elif isinstance(self.record, int):
            return [self.record]
        else:
            return self.record

    def plot(self, indices=None, cmap=None, refresh=None, showlast=None, redraw=True):
        lines = []
        inds = []
        if indices is None:
            recind = self.get_record_indices()
            for j, i in enumerate(recind):
                if cmap is None:
                    line, = pylab.plot(self.times, self[i], label=str(self.varname) + '[' + str(i) + ']')
                else:
                    line, = pylab.plot(self.times, self[i], label=str(self.varname) + '[' + str(i) + ']',
                               color=cmap(float(j) / (len(recind) - 1)))
                inds.append(i)
                lines.append(line)
        elif isinstance(indices, int):
            line, = pylab.plot(self.times, self[indices], label=str(self.varname) + '[' + str(indices) + ']')
            lines.append(line)
            inds.append(indices)
        else:
            for j, i in enumerate(indices):
                if cmap is None:
                    line, = pylab.plot(self.times, self[i], label=str(self.varname) + '[' + str(i) + ']')
                else:
                    line, = pylab.plot(self.times, self[i], label=str(self.varname) + '[' + str(i) + ']',
                               color=cmap(float(j) / (len(indices) - 1)))
                inds.append(i)
                lines.append(line)
        ax = pylab.gca()
        if refresh is not None:
            ylim = [Inf, -Inf]
            @network_operation(clock=EventClock(dt=refresh))
            def refresh_state_monitor_plot(clk):
                ymin, ymax = ylim
                if matplotlib.is_interactive():
                    if showlast is not None:
                        tmin = clk._t - float(showlast)
                        tmax = clk._t
                    for line, i in zip(lines, inds):
                        if showlast is None:
                            line.set_xdata(self.times)
                            y = self[i]
                        else:
                            imin = bisect.bisect_left(self.times, tmin)
                            imax = bisect.bisect_right(self.times, tmax)
                            line.set_xdata(self.times[imin:imax])
                            y = self[i][imin:imax]
                        line.set_ydata(y)
                        ymin = min(ymin, amin(y))
                        ymax = max(ymax, amax(y))
                    if showlast is None:
                        ax.set_xlim(0, clk._t)
                    else:
                        ax.set_xlim(clk._t - float(showlast), clk._t)
                    ax.set_ylim(ymin, ymax)
                    ylim[:] = [ymin, ymax]
                    if redraw:
                        pylab.draw()
            self.contained_objects.append(refresh_state_monitor_plot)

    def insert_spikes(self, spikemonitor, value=0):
        """
        Inserts spikes into recorded traces (for plotting). State values
        at spike times are replaced with the given value (peak value of spike).
        """
        dt = self.clock.dt
        values = self.values
        for i, neuron in enumerate(self.get_record_indices()):
            values[i,array(spikemonitor[neuron]/dt, dtype=int)] = value
        #self._values = values # or converted back to a list?


class RecentStateMonitor(StateMonitor):
    '''
    StateMonitor that records only the most recent fixed amount of time.
    
    Works in the same way as a :class:`StateMonitor` except that it has one
    additional initialiser keyword ``duration`` which gives the length of
    time to record values for, the ``record`` keyword defaults to ``True``
    instead of ``False``, and there are some different or additional
    attributes:
    
    ``values``, ``values_``, ``times``, ``times_``
        These will now return at most the most recent values over an
        interval of maximum time ``duration``. These arrays are copies,
        so for faster access use ``unsorted_values``, etc.
    ``unsorted_values``, ``unsorted_values_``, ``unsorted_times``, ``unsorted_times_``
        The raw versions of the data, the associated times may not be
        in sorted order and if ``duration`` hasn't passed, not all the
        values will be meaningful.
    ``current_time_index``
        Says which time index the next values to be recorded will be stored
        in, varies from 0 to M-1.
    ``has_looped``
        Whether or not the ``current_time_index`` has looped from M back to
        0 - can be used to tell whether or not every value in the
        ``unsorted_values`` array is meaningful or not (they will only all
        be meaningful when ``has_looped==True``, i.e. after time ``duration``).
    
    The ``__getitem__`` method also returns values in sorted order.
    
    To plot, do something like::
    
        plot(M.times, M.values[:, i])
    '''
    def __init__(self, P, varname, duration=5 * ms, clock=None, record=True, timestep=1, when='end'):
        StateMonitor.__init__(self, P, varname, clock=clock, record=record, timestep=timestep, when=when)
        self.duration = duration
        self.num_duration = int(duration / (timestep * self.clock.dt)) + 1
        if record is False:
            self.record_size = 0
        elif record is True:
            self.record_size = len(P)
        elif isinstance(record, int):
            self.record_size = 1
        else:
            self.record_size = len(record)
        self._values = zeros((self.num_duration, self.record_size))
        self._times = zeros(self.num_duration)
        self.current_time_index = 0
        self.has_looped = False
        self._invtargetdt = 1.0 / self.clock._dt
        self._arange = arange(len(P))

    def __call__(self):
        V = self.P.state_(self.varname)
        if self.record is False:
            self._mu += V
            self._sqr += V * V
        if self.record is not False and self.curtimestep == self.timestep:
            i = self._recordstep
            if self.record is not True:
                self._values[self.current_time_index, :] = V[self.record]
            else:
                self._values[self.current_time_index, :] = V
            self._times[self.current_time_index] = self.clock.t
            self._recordstep += 1
            self.current_time_index = (self.current_time_index + 1) % self.num_duration
            if self.current_time_index == 0: self.has_looped = True
        self.curtimestep -= 1
        if self.curtimestep == 0: self.curtimestep = self.timestep
        self.N += 1

    def __getitem__(self, i):
        timeinds = self.sorted_times_indices()
        if self.record is False:
            raise IndexError('Neuron ' + str(i) + ' was not recorded.')
        if self.record is not True:
            if isinstance(self.record, int) and self.record != i or (not isinstance(self.record, int) and i not in self.record):
                raise IndexError('Neuron ' + str(i) + ' was not recorded.')
            try:
                return self._values[timeinds, self.recordindex[i]]
            except:
                if i == self.record:
                    return self._values[timeinds, 0]
                else:
                    raise
        elif self.record is True:
            return self._values[timeinds, i]

    def get_past_values(self, times):
        # probably mostly to be used internally by Brian itself
        time_indices = (self.current_time_index - 1 - array(self._invtargetdt * asarray(times), dtype=int)) % self.num_duration
        if isinstance(times, SparseConnectionVector):
            return SparseConnectionVector(times.n, times.ind, self._values[time_indices, times.ind])
        else:
            return self._values[time_indices, self._arange]

    def get_past_values_sequence(self, times_seq):
        # probably mostly to be used internally by Brian itself
        if len(times_seq) == 0:
            return []
        time_indices_seq = [(self.current_time_index - 1 - array(self._invtargetdt * asarray(times), dtype=int)) % self.num_duration for times in times_seq]
        if isinstance(times_seq[0], SparseConnectionVector):
            return [SparseConnectionVector(times.n, times.ind, self._values[time_indices, times.ind]) for times, time_indices in izip(times_seq, time_indices_seq)]
        else:
            return [self._values[time_indices, self._arange] for times, time_indices in izip(times_seq, time_indices_seq)]

    def getvalues(self):
        return self._values
    getvalues_ = getvalues

    def sorted_times_indices(self):
        if not self.has_looped:
            return arange(self.current_time_index)
        return argsort(self._times)

    def get_sorted_times(self):
        return self._times[self.sorted_times_indices()]
    get_sorted_times_ = get_sorted_times

    def get_sorted_values(self):
        return self._values[self.sorted_times_indices(), :]
    get_sorted_values_ = get_sorted_values

    times = property(fget=get_sorted_times)
    times_ = property(fget=get_sorted_times_)
    values = property(fget=get_sorted_values)
    values_ = property(fget=get_sorted_values_)
    unsorted_times = property(fget=lambda self:array(self._times))
    unsorted_times_ = unsorted_times
    unsorted_values = property(fget=getvalues)
    unsorted_values_ = unsorted_values

    def reinit(self):
        # We check self._values is not None because the __init__ of this class
        # calls the __init__ of StateMonitor which calls reinit, but this happens
        # before self._values is set to be an array.
        if self._values is not None:
            self._values[:] = 0
            self._times[:] = 0
        self.current_time_index = 0
        self.N = 0
        self._recordstep = 0
        self._mu = zeros(len(self.P))
        self._sqr = zeros(len(self.P))
        self.has_looped = False

    def plot(self, indices=None, cmap=None, refresh=None, showlast=None, redraw=True):
        if refresh is not None and showlast is None:
            showlast = self.duration
        StateMonitor.plot(self, indices=indices, cmap=cmap, refresh=refresh, showlast=showlast, redraw=redraw)


class MultiStateMonitor(NetworkOperation):
    '''
    Monitors multiple state variables of a group
    
    This class is a container for multiple :class:`StateMonitor` objects,
    one for each variable in the group. You can retrieve individual
    :class:`StateMonitor` objects using ``M[name]`` or retrieve the
    recorded values using ``M[name, i]`` for neuron ``i``.

    Initialised with a group ``G`` and a list of variables ``vars``. If 
    ``vars`` is omitted then all the variables of ``G`` will be recorded.
    Any additional keyword argument used to initialise the object will
    be passed to the individual :class:`StateMonitor` objects (e.g. the
    ``when`` keyword).
    
    Methods:
    
    ``vars()``
        Returns the variables
    ``items()``, ``iteritems()``
        Returns the pairs (var, mon)
    ``plot([indices[, cmap]])``
        Plots all the monitors (note that real-time plotting is not supported
        for this class).
    
    Attributes:
    
    ``times``
        The times at which recordings were made.
    ``monitors``
        The dictionary of monitors indexed by variable name.
    
    Usage::
        
        G = NeuronGroup(N, eqs, ...)
        M = MultiStateMonitor(G, record=True)
        ...
        run(...)
        ...
        plot(M['V'].times, M['V'][0])
        figure()
        for name, m in M.iteritems():
            plot(m.times, m[0], label=name)
        legend()
        show()
    '''
    def __init__(self, G, vars=None, clock=None, **kwds):
        NetworkOperation.__init__(self, lambda : None, clock=clock)
        self.monitors = {}
        if vars is None:
            vars = [name for name in G.var_index.keys() if isinstance(name, str)]
        self.vars = vars
        for varname in vars:
            self.monitors[varname] = StateMonitor(G, varname, clock=clock, **kwds)
        self.contained_objects = self.monitors.values()

    def __getitem__(self, varname):
        if isinstance(varname, tuple):
            varname, i = varname
            return self.monitors[varname][i]
        else:
            return self.monitors[varname]

    def vars(self):
        return self.monitors.keys()

    def iteritems(self):
        return self.monitors.iteritems()

    def items(self):
        return self.monitors.items()

    def plot(self, indices=None, cmap=None):
        for k, m in self.monitors.iteritems():
            m.plot(indices, cmap=cmap)

    def get_times(self):
        return self.monitors.values()[0].times

    times = property(fget=lambda self:self.get_times())

    def __call__(self):
        pass


class CoincidenceCounter(SpikeMonitor):
    """
    Coincidence counter class.
    
    Counts the number of coincidences between the spikes of the neurons in the network (model spikes),
    and some user-specified data spike trains (target spikes). This number is defined as the number of 
    target spikes such that there is at least one model spike within +- ``delta``, where ``delta``
    is the half-width of the time window.
    
    Initialised as::
    
        cc = CoincidenceCounter(source, data, delta = 4*ms)
    
    with the following arguments:
    
    ``source``
        A :class:`NeuronGroup` object which neurons are being monitored.
    
    ``data``
        The list of spike times. Several spike trains can be passed in the following way.
        Define a single 1D array ``data`` which contains all the target spike times one after the
        other. Now define an array ``spiketimes_offset`` of integers so that neuron ``i`` should 
        be linked to target train: ``data[spiketimes_offset[i]], data[spiketimes_offset[i]+1]``, etc.
        
        It is essential that each spike train with the spiketimes array should begin with a spike at a
        large negative time (e.g. -1*second) and end with a spike that is a long time
        after the duration of the run (e.g. duration+1*second).
    
    ``delta=4*ms``
        The half-width of the time window for the coincidence counting algorithm.
    
    ``spiketimes_offset``
        A 1D array, ``spiketimes_offset[i]`` is the index of the first spike of 
        the target train associated to neuron i.
        
    ``spikedelays``
        A 1D array with spike delays for each neuron. All spikes from the target 
        train associated to neuron i are shifted by ``spikedelays[i]``.
        
    ``coincidence_count_algorithm``
        If set to ``'exclusive'``, the algorithm cannot count more than one
        coincidence for each model spike.
        If set to ``'inclusive'``, the algorithm can count several coincidences
        for a single model spike.
    
    ``onset``
        A scalar value in seconds giving the start of the counting: no
        coincidences are counted before ``onset``.
    
    Has three attributes:
    
    ``coincidences``
        The number of coincidences for each neuron of the :class:`NeuronGroup`.
        ``coincidences[i]`` is the number of coincidences for neuron i.
        
    ``model_length``
        The number of spikes for each neuron. ``model_length[i]`` is the spike
        count for neuron i.
        
    ``target_length``
        The number of spikes in the target spike train associated to each neuron.
    """
    def __init__(self, source, data=None, spiketimes_offset=None, spikedelays=None,
                 coincidence_count_algorithm='exclusive', onset=None, delta=4 * ms):

        source.set_max_delay(0)
        self.source = source
        self.delay = 0
        if onset is None:
            onset = 0 * ms
        self.onset = onset
        self.N = len(source)
        self.coincidence_count_algorithm = coincidence_count_algorithm

        self.data = array(data)
        if spiketimes_offset is None:
            spiketimes_offset = zeros(self.N, dtype='int')
        self.spiketimes_offset = array(spiketimes_offset)

        if spikedelays is None:
            spikedelays = zeros(self.N)
        self.spikedelays = array(spikedelays)

        dt = self.source.clock.dt
        self.delta = int(rint(delta / dt))
        self.reinit()

    def reinit(self):
        dt = self.source.clock.dt
        # Number of spikes for each neuron
        self.model_length = zeros(self.N, dtype='int')
        self.target_length = zeros(self.N, dtype='int')

        self.coincidences = zeros(self.N, dtype='int')
        self.spiketime_index = self.spiketimes_offset
        self.last_spike_time = array(rint(self.data[self.spiketime_index] / dt), dtype=int)
        self.next_spike_time = array(rint(self.data[self.spiketime_index + 1] / dt), dtype=int)

        # First target spikes (needed for the computation of 
        #   the target train firing rates)
        self.first_target_spike = zeros(self.N)

        self.last_spike_allowed = ones(self.N, dtype='bool')
        self.next_spike_allowed = ones(self.N, dtype='bool')

    def propagate(self, spiking_neurons):
        dt = self.source.clock.dt
        #T = array(rint((self.source.clock.t + self.spikedelays)/dt), dtype = int)
        spiking_neurons = array(spiking_neurons)
        if len(spiking_neurons):

            if self.source.clock.t >= self.onset:
                self.model_length[spiking_neurons] += 1

            T_spiking = array(rint((self.source.clock.t + self.spikedelays[spiking_neurons]) / dt), dtype=int)

            remaining_neurons = spiking_neurons
            remaining_T_spiking = T_spiking
            while True:
                remaining_indices, = (remaining_T_spiking > self.next_spike_time[remaining_neurons]).nonzero()
                if len(remaining_indices):
                    indices = remaining_neurons[remaining_indices]
                    self.target_length[indices] += 1
                    self.spiketime_index[indices] += 1
                    self.last_spike_time[indices] = self.next_spike_time[indices]
                    self.next_spike_time[indices] = array(rint(self.data[self.spiketime_index[indices] + 1] / dt), dtype=int)
                    if self.coincidence_count_algorithm == 'exclusive':
                        self.last_spike_allowed[indices] = self.next_spike_allowed[indices]
                        self.next_spike_allowed[indices] = True
                    remaining_neurons = remaining_neurons[remaining_indices]
                    remaining_T_spiking = remaining_T_spiking[remaining_indices]
                else:
                    break

            # Updates coincidences count
            near_last_spike = self.last_spike_time[spiking_neurons] + self.delta >= T_spiking
            near_next_spike = self.next_spike_time[spiking_neurons] - self.delta <= T_spiking
            last_spike_allowed = self.last_spike_allowed[spiking_neurons]
            next_spike_allowed = self.next_spike_allowed[spiking_neurons]
            I = (near_last_spike & last_spike_allowed) | (near_next_spike & next_spike_allowed)

            if self.source.clock.t >= self.onset:
                self.coincidences[spiking_neurons[I]] += 1

            if self.coincidence_count_algorithm == 'exclusive':
                near_both_allowed = (near_last_spike & last_spike_allowed) & (near_next_spike & next_spike_allowed)
                self.last_spike_allowed[spiking_neurons] = last_spike_allowed & -near_last_spike
                self.next_spike_allowed[spiking_neurons] = (next_spike_allowed & -near_next_spike) | near_both_allowed

class VanRossumMetric(StateMonitor):
    """
    van Rossum spike train metric.
    From M. van Rossum (2001): A novel spike distance (Neural Computation). 
    
    Compute the van Rossum distance between every spike train from the source
    population.
    
    Arguments:
    
    ``source``
        The group to compute the distances for. 
    ``tau``
        Time constant of the kernel (low pass filter).
    
    Has one attribute:
    
    ``distance``
        A square symmetric matrix containing the distances.    
    """
    def __init__(self, source, tau=2*ms):
        self.dt = source.clock.dt
        self.source = source
        self.nbr_neurons = len(source)
        self.tau=tau

        eqs="""
        dv/dt=(-v)/tau: volt
        """
        kernel=NeuronGroup(self.nbr_neurons,model=eqs)
        C = Connection(source, kernel, 'v')
        C.connect_one_to_one(source,kernel)
        StateMonitor.__init__(self,kernel, 'v', record=True)
        self.contained_objects=[kernel,C]
        #self.distance_matrix=zeros((self.nbr_neurons,self.nbr_neurons))  
               
    def reinit(self):
        StateMonitor.reinit(self)
        
#    def define(self):
#        @network_operation(clock=EventClock(dt=self.dt))
#        def get_distance_online():
#            tt=time()   
#            for neuron_idx1 in range(self.nbr_neurons):
#                for neuron_idx2 in range((neuron_idx1+1)):
#                    self.distance_matrix[neuron_idx1,neuron_idx2]=self.distance_matrix[neuron_idx1,neuron_idx2]+self.dt/self.tau*abs(self[neuron_idx1][-1]-self[neuron_idx2][-1])**2
#            print time()-tt
#        self.contained_objects.append(get_distance_online)
        
    def get_distance(self):

        if get_global_preference('useweave'):

            _cpp_compiler=get_global_preference('weavecompiler')
            _extra_compile_args=['-O3']
            if _cpp_compiler=='gcc':
                _extra_compile_args+=get_global_preference('gcc_options') # ['-march=native', '-ffast-math']        
            nbr_neurons=int(self.nbr_neurons)
            distance_matrix=zeros((nbr_neurons,nbr_neurons),dtype=float64)
            nbr_time_step=int(len(self[0]))
            dt=float(self.dt)
            traces=self.values
            tau=float(self.tau)
            
            code='''
            for(int k1=0;k1<nbr_neurons;k1++)
            {
                for(int k2=0;k2<k1;k2++)
                {
                    double &dm = distance_matrix[k1*nbr_neurons+k2];
                    double *tr1 = traces+k1*nbr_time_step;
                    double *tr2 = traces+k2*nbr_time_step;
                    for(int istep=0;istep<nbr_time_step;istep++, tr1++, tr2++)
                    {
                        double diff = *tr1-*tr2;
                        dm += diff*diff; 
                    }
                    dm *= dt/tau;
                }
            }
            '''
            tt=time() 
            weave.inline(code, ['nbr_time_step','nbr_neurons','dt','tau','distance_matrix','traces'],
                         compiler=_cpp_compiler,
                         extra_compile_args=_extra_compile_args)
            print time()-tt
            return tril(distance_matrix,k=0)+tril(distance_matrix,k=0).T
        else:
            nbr_time_step=int(len(self[0]))
            self.distance_matrix=zeros((self.nbr_neurons,self.nbr_neurons))
            values = self.values
            memsize_mb = float(self.nbr_neurons*nbr_time_step*8)/1024**2
           # tt=time()
            if memsize_mb>200:
                for neuron_idx1 in xrange(self.nbr_neurons):
                    vidx1 = values[neuron_idx1]
                    for neuron_idx2 in xrange((neuron_idx1+1)):
                        self.distance_matrix[neuron_idx1,neuron_idx2]=self.dt/self.tau*sum((vidx1-values[neuron_idx2])**2)
            else:
                for neuron_idx1 in xrange(self.nbr_neurons):
                    Vi = values[neuron_idx1].reshape((1, nbr_time_step))
                    Vj = values.reshape((self.nbr_neurons, nbr_time_step))
                    self.distance_matrix[neuron_idx1, :] = (self.dt/self.tau)*sum((Vi-Vj)**2, axis=1)                
            #print time()-tt

            return tril(self.distance_matrix,k=0)+tril(self.distance_matrix,k=0).T

            
    distance = property(fget=get_distance)



class CoincidenceMatrixCounter(SpikeMonitor):
    """ 
    Coincidence counter matrix class.
    
    Counts the number of coincidences between the spikes of the neurons in the network (model spikes). This yields a matrix
    with the coincidence counts between every pair of neurons in the network
    
    Initialised as::
    
        cc = CoincidenceCounter(source, delta = 4*ms)
    
    with the following arguments:
    
    ``source``
        A :class:`NeuronGroup` object which neurons are being monitored.
        
    ``delta=4*ms``
        The half-width of the time window for the coincidence counting algorithm.      

    ``onset``
        A scalar value in seconds giving the start of the counting: no
        coincidences are counted before ``onset``.
    
    Has three attributes:
    
    ``coincidences``
        The matrix containg the number of coincidences between each neuron of the :class:`NeuronGroup`.
        ``coincidences[i,j]`` is the number of coincidences between neuron i and j.
        
    ``model_length``
        The number of spikes for each neuron. ``model_length[i]`` is the spike
        count for neuron i.
    
    """
    def __init__(self, source, onset=None, delta=4 * ms):

        source.set_max_delay(0)
        self.source = source
        self.delay = 0
        if onset is None:
            onset = 0 * ms
        self.onset = onset
        self.N = len(source)

        dt = self.source.clock.dt
        self.delta = array(rint(delta / dt), dtype=int)
        self.reinit()

    def reinit(self):
        dt = self.source.clock.dt # does not seem to be used
        # Number of spikes for each neuron
        self.model_length = zeros(self.N, dtype='int')
        self.target_length = zeros(self.N, dtype='int')

        self._coincidences = zeros((self.N, self.N), dtype='int')
        self.last_spike_time = -100 * ones(self.N, dtype='int')

    def get_coincidences(self):
        M = array(self._coincidences, dtype=float)
        M -= diag(M.diagonal() / 2)
        return M

    coincidences = property(fget=get_coincidences)

    def propagate(self, spiking_neurons):
        dt = self.source.clock.dt
        spiking_neurons = array(spiking_neurons)
        if len(spiking_neurons):
            if self.source.clock.t >= self.onset:
                self.model_length[spiking_neurons] += 1
            tint = array(rint(self.source.clock.t / dt), dtype=int)
            self.last_spike_time[spiking_neurons] = tint

            I, = (abs(self.last_spike_time - tint) <= self.delta).nonzero()
            if self.source.clock.t >= self.onset:
                for ispike in spiking_neurons:
                    #self.coincidences[ispike,setdiff1d(I,ispike)]+=1
                    self._coincidences[ispike, I[ispike <= I]] += 1
                    self._coincidences[I[ispike <= I], ispike] += 1
#        if self.source.clock.t == self.onset:
#            self.coincidences=(self.coincidences+self.coincidences.T)/2


class StateHistogramMonitor(NetworkOperation, Monitor):
    '''
    Records the histogram of a state variable from a :class:`NeuronGroup`.

    Initialise as::
    
        StateHistogramMonitor(P,varname,range(,period=1*ms)(,nbins=20))
    
    Where:
    
    ``P``
        The group to be recorded from
    ``varname``
        The state variable name or number to be recorded
    ``range``
        The minimum and maximum values for the state variable. A 2-tuple of floats.
    ``period``
        When to record.
    ``nbins``
        Number of bins for the histogram.

    The :class:`StateHistogramMonitor` object has the following properties:

    ``mean``
        The mean value of the state variable for every neuron in the
        group
    ``var``
        The unbiased estimate of the variances, as in ``mean``
    ``std``
        The square root of ``var``, as in ``mean``
    ``hist``
        A 2D array of the histogram values of all the neurons, each row is a
        single neuron's histogram.
    ``bins``
        A 1D array of the bin centers used to compute the histogram
    ``bin_edges``
        A 1D array of the bin edges used to compute the histogram
        
    In addition, if :class:`M`` is a :class:`StateHistogramMonitor` object, you write::
    
        M[i]
    
    for the histogram of neuron ``i``.
    '''
    def __init__(self, group, varname, range, period=1 * ms, nbins=20):
        self.clock = Clock(period)
        NetworkOperation.__init__(self, None, clock=self.clock)
        self.group = group
        self.len = len(group)
        self.varname = varname
        self.nbins = nbins
        self.n = 0
        self.bin_edges = linspace(range[0], range[1], self.nbins + 1)
        self._hist = zeros((self.len, self.nbins + 2))
        self.sum = zeros(self.len)
        self.sum2 = zeros(self.len)

    def __call__(self):
        x = self.group.state_(self.varname)
        self.sum += x
        self.sum2 += x ** 2
        inds = digitize(x, self.bin_edges)
        for i in xrange(self.len):
            self._hist[i, inds[i]] += 1
        self.n += 1

    def __getitem__(self, i):
        return self.hist[i, :]

    hist = property(fget=lambda self:self._hist[:, 1:-1] / self.n)
    bins = property(fget=lambda self:(self.bin_edges[:-1] + self.bin_edges[1:]) / 2)

    mean = property(fget=lambda self:self.sum / self.n)
    var = property(fget=lambda self:(self.sum2 / self.n - self.mean ** 2))
    std = property(fget=lambda self:self.var ** .5)
