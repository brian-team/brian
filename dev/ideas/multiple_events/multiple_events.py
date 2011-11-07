"""
Here I try do define multiple events on a NeuronGroup (not just spikes).
"""
from brian import *
import copy

#class ParasiteGroup(NeuronGroup):
#    def __init__(self, P, threshold, reset=NoReset()):
#        # A group that tracks other threshold events
#        self = copy.copy(P)
#        self._owner=self # This separates it from its parents
#        self.set_instance_id(level=0) # This makes available to magic tools
#        self._state_updater=LazyStateUpdater() # Does not update the state matrix
#        self._threshold=StringThreshold(threshold, level=0) # Defines a new threshold
#        self._resetfun=reset # No reset
#        self.set_max_delay(self._max_delay*defaultclock.dt) # New LS container

if __name__=='__main__':
    tau=10*ms
    P=NeuronGroup(10,model='dv/dt=(1.1-v)/tau:1',threshold=1,reset=0)
    #P2=ParasiteGroup(P,"v>0.8")

    P2 = copy.copy(P)
    P2._owner=P2 # This separates it from its parents
    P2.set_instance_id(level=0) # This makes available to magic tools
    P2._state_updater=LazyStateUpdater() # Does not update the state matrix
    P2._threshold=StringThreshold("v>0.8", level=0) # Defines a new threshold
    P2._resetfun=NoReset() # No reset
    P2.set_max_delay(P2._max_delay*defaultclock.dt) # New LS container
    
    M=StateMonitor(P,'v',record=0)
    S=SpikeMonitor(P)
    S2=SpikeMonitor(P2)
    run(100*ms)
    subplot(311)
    plot(M.times/ms,M[0])
    subplot(312)
    raster_plot(S)
    subplot(313)
    raster_plot(S2)
    show()
    