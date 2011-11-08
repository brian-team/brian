"""
Here I try do define multiple events on a NeuronGroup (not just spikes).

How to make this in a nice way?
1) Insert it in NeuronGroup. We would pass a list of threshold/reset. It is
   handled as currently.
2) Have multiple states, as we thought for refractoriness. But instead of
   having one system of equations for each state, we just have a separate
   threshold and reset.
3a) Insert it in NeuronGroup, but have a new integer variable, "state". Each
   state corresponds to a specific additional event type. Additional threshods/
   resets are called only when the neuron is in state 0, 1, etc. Ex:
     threshold=("v>1","v>0.8","v<0.8") # on state: all, 0, 1
     reset=("v=0","state=1","state=0")
 b) This could also be implicit, that is, we assume that event triggering is
   prevented as long as the condition still holds true. This would give simply:
     threshold=("v>1","v>0.8")
     reset="v=0"
   and the state would be automatically dealt with, possibly via a new threshold
   class, e.g. PositiveCrossing("v>1").
4) Have a new class attached to a group, e.g.:
   ParasiteGroup(P,threshold,reset)
   Threshold for that class could, by default, be a positive crossing.
   This could create a subgroup of P.
   
I prefer 3 & 4. Option 4 is simpler, but perhaps might integrate less
nicely with STDP (but not sure). But it also works nicely with SpikeMonitor.

Here option 4 is implemented.
"""
from brian import *
from brian.utils.circular import *
import copy

class PositiveCrossing(Threshold):
    '''
    The event (spike) is triggered only the first time
    the threshold condition is satisfied. That is, after an event is triggered,
    further events are prevented until the condition is false again.

    **Initialised as:** ::
    
        PositiveCrossing(threshold)

    where ``threshold`` is another threshold.
    
    Warning: an instance can be used with only one group.
    '''
    # Not the most efficient way to do it, but it works!
    def __init__(self,threshold):
        self._threshold=threshold
        self._firsttime=True

    def __call__(self, P):
        if self._firsttime:
            self._firsttime=False
            self._available=ones(len(P),dtype=bool_)
        all_spikes=self._threshold(P)
        spikes=all_spikes[self._available[all_spikes]]
        self._available[:]=True
        self._available[all_spikes]=False
        return spikes

    def __repr__(self):
        return "Positive crossing threshold"
    
def ParasiteGroup(P, threshold, reset=NoReset()):
    '''
    This is a NeuronGroup that tracks events on P, that is,
    that produces a spike when the threshold condition is true and was
    false at the previous time step (positive crossing).
    
    Optionally, a reset operation can be applied (maybe we could
    remove this).
    
    Events are stored for as long as spikes in P (could be an option).
    '''
    # This would be probably nicer in a proper subclass
    # or WatcherGroup?
    Q=P[:] # New subgroup
    Q._owner=Q # This separates it from its parents (otherwise it is not run)
    Q.set_instance_id(level=1) # This makes available to magic tools
    Q._state_updater=LazyStateUpdater() # Does not update the state matrix
    Q._threshold=PositiveCrossing(StringThreshold(threshold, level=1)) # Defines a new threshold
    Q._resetfun=reset # No reset
    Q.LS = SpikeContainer(Q._max_delay,
                          useweave=get_global_preference('useweave'),
                          compiler=get_global_preference('weavecompiler')) # Spike storage
    return Q
    
if __name__=='__main__':
    tau=10*ms
    P=NeuronGroup(10,model='dv/dt=(1.1-v)/tau:1',threshold=1,reset=0)
    P2=ParasiteGroup(P,"v>0.6")
    P3=ParasiteGroup(P,"v>0.8")
    
    M=StateMonitor(P,'v',record=0)
    S=SpikeMonitor(P)
    S2=SpikeMonitor(P2)
    S3=SpikeMonitor(P3)
    run(100*ms)
    subplot(411)
    plot(M.times/ms,M[0])
    subplot(412)
    raster_plot(S)
    xlim(0,100)
    subplot(413)
    raster_plot(S2)
    xlim(0,100)
    subplot(414)
    raster_plot(S3)
    xlim(0,100)
    show()
    