import brian.experimental.cspikequeue.cspikequeue as _cspikequeue
from brian.monitor import SpikeMonitor
from brian.stdunits import ms
import warnings
import numpy as np
import sys

__all__=['CSpikeQueueWrapper']
INITIAL_MAXSPIKESPER_DT = 1

class CSpikeQueueWrapper(SpikeMonitor):
    def __init__(self, source, synapses, delays,
                 max_delay = 60*ms, maxevents = INITIAL_MAXSPIKESPER_DT,
                 precompute_offsets = True):
        super(CSpikeQueueWrapper, self).__init__(source, 
                                                 record = False)

        nsteps = int(np.floor(max_delay/self.source.clock.dt))+1
        print "Initialized with %d steps and %d events" % (nsteps, maxevents)

        self._max_delay = max_delay
        
        self.synapses = synapses
        self.delays = delays # Delay handling should also be in C

        self._spikequeue = _cspikequeue.SpikeQueue(nsteps, int(maxevents))

    def compress(self):
        nsteps=max(self.delays)+1

        # Check whether some delays are too long
        if (nsteps>self._spikequeue.n_delays):
            desired_max_delay = nsteps * self.source.clock.dt
            print desired_max_delay
            raise ValueError,"Synaptic delays exceed maximum delay, set max_delay to %.1f ms" % (desired_max_delay/ms)
        
        if hasattr(self, '_iscompressed') and self._iscompressed:
            return
        self._iscompressed = True
        
        # Adjust the maximum delay and number of events per timestep if necessary
        maxevents=self._spikequeue.n_maxevents
        if maxevents==INITIAL_MAXSPIKESPER_DT: # automatic resize
            maxevents=max(INITIAL_MAXSPIKESPER_DT, max([len(targets) for targets in self.synapses]))

        # Resize
        self._spikequeue.expand(int(maxevents))# Resize


    def insert(self, *args, **kargs):
        self._spikequeue.insert(*args)

    def next(self):
        return self._spikequeue.next()

    def peek(self):
        return self._spikequeue.peek()
    

