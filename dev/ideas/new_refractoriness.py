"""
Idea:

Refractoriness should be controlled by allowing user provided code for any of
the following events:

* Start of refractoriness (equivalent to reset in the old scheme)
* During refractoriness (equivalent to refractoriness in the old scheme)
* End of refractoriness (didn't exist in the old scheme)

Code can set any of the following variables:

* is_refractory: directly control whether or not the neuron is in a refractory
  period. If it is set False during refractoriness it will automatically trigger
  the end of refractoriness event.
* refractory_until, refractory_for: control the duration of the refractoriness
  either by setting the end time, or the duration
  
In addition, it is backwards compatible in that you can specify an array of
refractory times, or a state variable which can be used for variable
refractoriness, or of course just a single value.

Users provide code either as Python functions or strings for each of the three
events, but there are some default or built-in types of refractoriness you can
use, notably the reset and clamp. The reset will allow you to just specify a
reset function (like V=Vr) and will set the refractory_for based on either the
constant, array or state variable. The clamp automatically records the
post-reset values of one or more variables and holds them there for the period
of the refractoriness.

Internally, it is implemented using the existing scheme of having an internal
NeuronGroup variable _next_allowed_spiketime which gives the time when the
neuron is next allowed to spike. Each time step, the following sequence of
operations is carried out:

* spikes = threshold()
* Filter spikes based on t>_next_allowed_spiketime
* Trigger during_refractoriness events based on _next_allowed_spiketime>=t
* Trigger end_refractoriness events based on _next_allowed_spiketime<t+dt
* Trigger start_refractoriness events based on filtered spikes

The reason for this sequence of operations is that we do not want to follow
a start_refractoriness with a during_refractoriness, and we want to allow a
during_refractoriness to trigger an end_refractoriness.

In terms of efficiency, checking t>_next_allowed_spiketime is O(num neurons) but
I don't see any way to avoid this if we want to allow variable refractoriness,
and in any case the state update and threshold are already O(num neurons). We
could have optimisations in the case that we know that user code doesn't use the
variable refractoriness features.

TODO:

* Nice syntax for users
* Allow strings for code
  + Python and C versions preferably
"""

from brian import *
import numpy

class TriggeringArray(numpy.ndarray):
    def __new__(subtype, arr, trigger):
        # All numpy.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        return numpy.array(arr, copy=False).view(subtype)
    def __init__(self, arr, trigger):
        self.trigger = trigger
        self.triggered = False
    def __getitem__(self, item):
        return asarray(numpy.ndarray.__getitem__(self, item))
    def __setitem__(self, item, value):
        self.trigger(item, value)
        self.triggered = True
        numpy.ndarray.__setitem__(self, item, value)

OrigNeuronGroup = NeuronGroup
class NeuronGroup(OrigNeuronGroup):
    def __init__(self, *args, **kwds):
        refractoriness = kwds.pop('refractoriness')
        OrigNeuronGroup.__init__(self, *args, **kwds)
        self._refractoriness = refractoriness
        self.is_refractory = TriggeringArray(zeros(len(self), dtype=bool),
                                             self._is_refractory_trigger)
        self.refractory_until = TriggeringArray(zeros(len(self)),
                                                self._refractory_until_trigger)
        self.refractory_for = TriggeringArray(zeros(len(self)),
                                              self._refractory_for_trigger)

    def _is_refractory_trigger(self, spikes, values):
        if isinstance(values, numpy.ndarray):
            self._next_allowed_spiketime[spikes[values!=0]] = 1e300 # i.e. refractory for ever
            self._next_allowed_spiketime[spikes[values==0]] = self.clock._t
        else:
            if values:
                self._next_allowed_spiketime[spikes] = 1e300
            else:
                self._next_allowed_spiketime[spikes] = self.clock._t
    
    def _refractory_for_trigger(self, spikes, values):
        self._next_allowed_spiketime[spikes] = self.clock._t+values
    
    def _refractory_until_trigger(self, spikes, values):
        self._next_allowed_spiketime[spikes] = values
        
    def update(self):
        self._state_updater(self) # update the variables
        if self._spiking:
            refrac = self._refractoriness
            spikes = self._threshold(self) # get spikes
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            # Filter refractory spikes
            spikes = spikes[self._next_allowed_spiketime[spikes]<=self.clock._t]
            # Trigger during_refractory based on currently refractory indices
            current_refrac_time = self._next_allowed_spiketime-self.clock._t
            currently_refractory, = (current_refrac_time>=0).nonzero()
            if refrac.during is not None:
                refrac.during(self, currently_refractory)
                # might need to recompute this if is_refractory or others have
                # been used.
                if self.is_refractory.triggered or self.refractory_until.triggered or self.refractory_for.triggered:
                    self.is_refractory.triggered = False
                    self.refractory_until.triggered = False
                    self.refractory_for.triggered = False
                    current_refrac_time[currently_refractory] = self._next_allowed_spiketime[currently_refractory]-self.clock._t
            # Trigger end_refractory based on stop_refractory
            stop_refractory = currently_refractory[current_refrac_time[currently_refractory]<self.clock._dt]
            if refrac.end is not None:
                refrac.end(self, stop_refractory)
            self._next_allowed_spiketime[stop_refractory] = self.clock._t
            # Trigger start_refractory based on spikes array
            if refrac.start is not None:
                refrac.start(self, spikes)
            # Store spikes
            self.LS.push(spikes)

def variable_refractory_start(group, spikes):
    if group._variable_refractory_time:
        if group._refractory_variable is not None:
            refractime = group.state_(group._refractory_variable)
        else:
            refractime = group._refractory_array
        group._next_allowed_spiketime[spikes] = group.clock._t+refractime[spikes]
    else:
        group._next_allowed_spiketime[spikes] = group.clock._t+group._refractory_time

class RefractoryReset(object):
    def __init__(self, reset):
        self.reset = reset
    def __call__(self, group, spikes):
        self.reset(group, spikes)
        variable_refractory_start(group, spikes)

class RefractoryClamp(object):
    def __init__(self, start=None, vars=None):
        self.additional_start = start
        if vars is None:
            vars = (0,)
        self.vars = vars
    def start(self, group, spikes):
        if self.additional_start is not None:
            self.additional_start(group, spikes)
        if not hasattr(self, 'clamped_values'):
            self.clamped_values = dict((var, zeros(len(group))) for var in self.vars)
        for var, vals in self.clamped_values.iteritems():
            vals[spikes] = group.state_(var)[spikes]
    def during(self, group, spikes):
        if not hasattr(self, 'clamped_values'):
            return
        for var, vals in self.clamped_values.iteritems():
            group.state_(var)[spikes] = vals[spikes]

class Refractoriness(object):
    def __init__(self, start=None, during=None, end=None):
        self.start = start
        self.during = during
        self.end = end

if __name__=='__main__':
    eqs = '''
    dV/dt = (2-V)/(10*ms) : 1
    dw/dt = -w/(10*ms) : 1
    drefractory/dt = -refractory/(250*ms) : second
    '''
    if 0:
        def reset(G, spikes):
            G.V[spikes] = 0
        start = RefractoryReset(reset)
        clamp = RefractoryClamp(start, ('V', 'w'))
        start = clamp.start
        during = clamp.during
        def end(G, indices):
            G.w[indices] += 1
    if 1:
        def start(G, spikes):
            G.V[spikes] = 0
            G.w[spikes] += G.refractory[spikes]
            G.is_refractory[spikes] = True
            #G.refractory_for[spikes] = G.refractory[spikes]
            #G.refractory_until[spikes] = 10*ms+G.refractory[spikes]
        def during(G, indices):
            G.V[indices] = 0
            G.is_refractory[indices] = G.w[indices]>1*ms
        def end(G, indices):
            G.w[indices] = 0
    ref = Refractoriness(start, during, end)
    G = NeuronGroup(3, eqs, threshold=1, refractoriness=ref,
                    refractory='refractory',
                    max_refractory=5*ms # dummy value
                    )
    G._variable_refractory_time = True
    G._refractory_variable = 'refractory'
    G._refractory_array = None
    G.refractory = [2*ms, 3*ms, 5*ms]
    M = MultiStateMonitor(G, record=True)
    Msp = SpikeMonitor(G)
    run(40*ms)
    for k in Msp.spikes:
        print k
    for i, var in enumerate(M.vars):
        subplot(len(M.vars), 1, i+1)
        M[var].plot()
        title(var)
    show()
    