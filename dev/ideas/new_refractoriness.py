"""
Idea:

Layer 1: is_refractory array, spikes are checked against this
         - User can set is_refractory by hand
         - Automatically maintain a list currently_refractory.
           Use computed is_refractory attributes to update this.
Layer 2: Event based stop_refractory, sets is_refractory=0
         - Spike container can be used, but dynamically resize
           the circular array of future spike times when
           necessary, as we are putting events into future states
           rather than past states, so we don't need to know
           max_refractory for this
         - Can set either refractory_until=time or refractory_for=time
           and it automatically inserts the event at the appropriate
           time
Layer 3: Value for each neuron, refractory=time will automatically insert
         layer 2 events.
         
Events:

- start_refractory, reset: triggered by threshold
- during_refractory: triggered for all currently_refractory indices each step
- end_refractory: triggered by is_refractory=0

Sequence of operations:

* Threshold gives spikes array
* Reduce spikes array by filtering out neurons which are refractory
* Trigger during_refractory based on currently_refractory
* Trigger end_refractory based on stop_refractory (although this could
  be defined by setting is_refractory=0 or other mechanisms)
* Trigger start_refractory based on spikes array

New thoughts:

* Calendar queue is not nice because we want to insert events with different
  delays. Instead, keep using the next spiketime thing in NeuronGroup but
  update it so that start, during and end events can be extracted from it.
"""

from brian import *
import numpy

OrigNeuronGroup = NeuronGroup

class NeuronGroup(OrigNeuronGroup):
    def __init__(self, *args, **kwds):
        refractoriness = kwds.pop('refractoriness')
        OrigNeuronGroup.__init__(self, *args, **kwds)
        self._refractoriness = refractoriness
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
    dw/dt = -w/(50*ms) : 1
    drefractory/dt = -refractory/(250*ms) : second
    '''
    def reset(G, spikes):
        G.V[spikes] = 0
    start = RefractoryReset(reset)
    clamp = RefractoryClamp(start, ('V', 'w'))
    start = clamp.start
    during = clamp.during
    def end(G, indices):
        G.w[indices] += 1
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
        subplot(1, len(M.vars), i+1)
        M[var].plot()
        title(var)
    show()
    