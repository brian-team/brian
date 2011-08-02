'''
Simple implementation of an event scheduling scheme for Brian.

The idea is that a spike schedules a future event V[i] += val for i an index
and val a float value. Below this idea is used to implement dendritic delays
in an alternative way to DelayConnection. You could implement axonal delays by
using V[i] += W[j] for i and j both indices (here j would be a pointer to an
appropriate entry in a weight matrix W, and i an appropriate neuron state
variable index).
'''

from brian import *
from scipy import weave
import numpy
from itertools import izip

N = 10
max_events_per_timestep = 100000

G = NeuronGroup(1, 'V:1', reset=0, threshold=0.5)
H = NeuronGroup(N, 'V:1', reset=0, threshold=0.5)


class ScheduledEventDelayConnection(DelayConnection):
    def __init__(self, *args, **kwds):
        DelayConnection.__init__(self, *args, **kwds)
        self.scheduled_events_ptr = numpy.zeros((self._max_delay, max_events_per_timestep), dtype=int)
        self.scheduled_events_val = numpy.zeros((self._max_delay, max_events_per_timestep), dtype=float)
        self.num_scheduled_events = numpy.zeros(self._max_delay, dtype=int)
        self.scheduled_events_index = 0
        @network_operation(when='after_connections') # for the moment, don't worry about clock details
        def process_scheduled_events():
            num_scheduled_events = int(self.num_scheduled_events[self.scheduled_events_index])
            if num_scheduled_events:
                ptr = self.scheduled_events_ptr[self.scheduled_events_index, :num_scheduled_events]
                val = self.scheduled_events_val[self.scheduled_events_index, :num_scheduled_events]
                sv = self.target._S[self.nstate]
                #sv[ptr] += val
                I = argsort(ptr)
                ptrs = ptr[I]
                vals = val[I]
                J = hstack((True, (ptrs[1:]!=ptrs[:-1]))).nonzero()[0]
                sv[ptrs[J]] += add.reduceat(vals, J)
                # with a bit of work, this can be made to work for any ufunc using
                # the .reduceat method of ufuncs after sorting ptr, etc. Still doesn't work
                # in the general case of any Python expression though.
                self.num_scheduled_events[self.scheduled_events_index] = 0
            self.scheduled_events_index = (self.scheduled_events_index + 1) % self._max_delay
        self.contained_objects.append(process_scheduled_events)
        self.constant_delays = True # set to True to assume that delays will not change over time
        self.constant_weights = True
        self.need_to_precompute_sorted_delays = self.constant_delays
        for f in self.contained_objects:
            if f.__name__=='delayed_propagate':
                self.contained_objects.remove(f)
                break

    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if self.need_to_precompute_sorted_delays:
            self.sort_indices = []
            self.int_delay = []
            self.target_indices = []
            self.sorted_Wrow = []
            self.Jp = []
            self.event_index_offset = []
            self.added_events = []
            for i in xrange(len(self.source)):
                dvecrow = self.delayvec[i, :]
                int_delay = array(self._invtargetdt * dvecrow, dtype=int)
                sort_indices = argsort(int_delay)
                int_delay = int_delay[sort_indices]
                J = int_delay[1:] != int_delay[:-1]
                K = int_delay[1:] == int_delay[:-1]
                A = hstack((0, cumsum(array(J, dtype=int))))
                B = hstack((0, cumsum(array(K, dtype=int))))
                BJ = hstack((0, B[J]))
                event_index_offset = B - BJ[A]
                Jp = hstack((J, True))
                added_events = event_index_offset[Jp] + 1
                self.sort_indices.append(sort_indices)
                self.int_delay.append(int_delay)
                self.event_index_offset.append(event_index_offset)
                self.Jp.append(Jp)
                self.target_indices.append(dvecrow.ind[sort_indices])
                self.added_events.append(added_events)
                if self.constant_weights:
                    self.sorted_Wrow.append(self.W[i, :][sort_indices])
            self.need_to_precompute_sorted_delays = False
        if len(spikes):
            if False:
                # C code version
                nspikes = len(spikes)
                W = asarray(self.W)
                n, m = W.shape
                numevents = self.num_scheduled_events
                ptr, val = self.scheduled_events_ptr, self.scheduled_events_val
                index = self.scheduled_events_index
                idt = self._invtargetdt
                md = self._max_delay
                dvec = asarray(self.delayvec)
                sv = self.target._S[self.nstate]
                code = '''
                for(int i=0; i<nspikes; i++)
                {
                    int k = spikes(i);
                    for(int j=0; j<m; j++)
                    {
                        int queue_index = (index+(int)(idt*dvec(k, j)))%md;
                        int event_index = numevents(queue_index)++;
                        ptr(queue_index, event_index) = j;
                        val(queue_index, event_index) = W(k, j);
                    }
                }
                '''
                weave.inline(code, ['numevents', 'ptr', 'val', 'nspikes', 'n', 'm', 'index', 'idt',
                                    'md', 'dvec', 'sv', 'spikes', 'W'],
                             type_converters=weave.converters.blitz,
                             compiler='gcc')
            else:
                # Python code only version
                ptr, val = self.scheduled_events_ptr, self.scheduled_events_val
                #W = asarray(self.W)
                Wrows = self.W.get_rows(spikes)
                dvecrows = self.delayvec.get_rows(spikes)
                num_scheduled_events = self.num_scheduled_events
                sparse = isinstance(dvecrows[0], SparseConnectionVector)
                _max_delay = self._max_delay
                scheduled_events_index = self.scheduled_events_index
                precomp = self.constant_delays
                precomp_weights = self.constant_weights
                for i, dvecrow, Wrow in izip(spikes, dvecrows, Wrows):
                    dvecrowarr = asarray(dvecrow)
                    if precomp:
                        int_delay = self.int_delay[i]
                        sort_indices = self.sort_indices[i]
                        event_index_offset = self.event_index_offset[i]
                        Jp = self.Jp[i]
                        added_events = self.added_events[i]
                    else:
                        int_delay = array(self._invtargetdt * dvecrowarr, dtype=int)
                        sort_indices = argsort(int_delay)
                        int_delay = int_delay[sort_indices]
                        J = int_delay[1:] != int_delay[:-1]
                        K = int_delay[1:] == int_delay[:-1]
                        A = hstack((0, cumsum(array(J, dtype=int))))
                        B = hstack((0, cumsum(array(K, dtype=int))))
                        BJ = hstack((0, B[J]))
                        event_index_offset = B - BJ[A]
                        Jp = hstack((J, True))
                        added_events = event_index_offset[Jp] + 1
                    queue_index = (scheduled_events_index + int_delay) % _max_delay
                    event_index = event_index_offset + num_scheduled_events[queue_index]
                    if sparse:
                        if precomp:
                            target_indices = self.target_indices[i]
                        else:
                            target_indices = dvecrow.ind[sort_indices]
                    else:
                        target_indices = sort_indices
                    ptr[queue_index, event_index] = target_indices
                    if precomp_weights:
                        sorted_Wrow = self.sorted_Wrow[i]
                    else:
                        sorted_Wrow = Wrow[sort_indices]
                    val[queue_index, event_index] = sorted_Wrow
                    num_scheduled_events[queue_index[Jp]] += added_events

if 0:
    C = ScheduledEventDelayConnection(G, H, structure='dense', max_delay=1 * ms)
    C.connect_full(G, H, weight=1, delay=(0 * ms, 1 * ms))
    
    G.spikemon = SpikeMonitor(G)
    H.spikemon = SpikeMonitor(H)
    G.statemon = MultiStateMonitor(G, record=True)
    H.statemon = MultiStateMonitor(H, record=True)
    
    G.V = 1
    
    run(1.2 * ms)
    
    print 'H.spikemon.nspikes', H.spikemon.nspikes
    print H.spikemon.spikes
    print asarray(C.delay) / ms
    
    subplot(221)
    raster_plot(G.spikemon)
    subplot(222)
    raster_plot(H.spikemon)
    subplot(223)
    G.statemon.plot()
    subplot(224)
    H.statemon.plot()
    show()
else:
    # speed test
    import time, gc, random, numpy
    do_profile = False
    set_global_preferences(useweave=False, usenewpropagate=False)
    for name, Conn in [
                       ('Normal', DelayConnection),
                       ('Events', ScheduledEventDelayConnection),
                       ]:
        clear(True, True)
        reinit_default_clock()
        gc.collect()
        random.seed(2134234)
        numpy.random.seed(324342)
        eqs = '''
        dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
        dge/dt = -ge/(5*ms) : volt
        dgi/dt = -gi/(10*ms) : volt
        '''
        
        P = NeuronGroup(4000, model=eqs,
                      threshold= -50 * mV, reset= -60 * mV)
        P.v = -60 * mV + 10 * mV * rand(len(P))
        Pe = P.subgroup(3200)
        Pi = P.subgroup(800)
        
        Ce = Conn(Pe, P, 'ge', weight=1.62 * mV, sparseness=0.02, delay=(0*ms, 5*ms), structure='sparse')
        Ci = Conn(Pi, P, 'gi', weight= -9 * mV, sparseness=0.02, delay=(0*ms, 5*ms), structure='sparse')

        M = SpikeCounter(P)
        
        run(1*ms)
        
        start = time.time()
        
        try:
            if do_profile:
                import cProfile as profile
                import pstats
                profile.run('run(.1*second)','scheduled_events.prof')
                stats = pstats.Stats('scheduled_events.prof')
                stats.strip_dirs()
                stats.sort_stats('cumulative', 'calls')
                stats.print_stats(50)
                import os
                os.remove('scheduled_events.prof')
            else:
                run(.1*second)
        except IndexError:
            print sum(M.count)
            raise
        
        print name, sum(M.count), time.time()-start
