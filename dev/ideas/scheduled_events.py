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

N = 10
max_events_per_timestep = 100

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
                J = unique(ptr)
                K = digitize(ptr, J) - 1
                b = bincount(K)
                sv[J] += b * val
                # with a bit of work, this can be made to work for any ufunc using
                # the .reduceat method of ufuncs after sorting ptr, etc. Still doesn't work
                # in the general case of any Python expression though.
                self.num_scheduled_events[self.scheduled_events_index] = 0
            self.scheduled_events_index = (self.scheduled_events_index + 1) % self._max_delay
        self.contained_objects.append(process_scheduled_events)

    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
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
                W = asarray(self.W)
                for i in spikes:
#                    print 'num_scheduled_events before', self.num_scheduled_events
                    dvecrow = self.delayvec[i, :]
                    int_delay = numpy.array(self._invtargetdt * dvecrow, dtype=int)
                    sort_indices = argsort(int_delay)
                    int_delay = int_delay[sort_indices]
                    queue_index = (self.scheduled_events_index + int_delay) % self._max_delay
                    J = int_delay[1:] != int_delay[:-1]
                    K = int_delay[1:] == int_delay[:-1]
                    A = hstack((0, cumsum(array(J, dtype=int))))
                    B = hstack((0, cumsum(array(K, dtype=int))))
                    BJ = hstack((0, B[J]))
                    event_index_offset = B - BJ[A]
                    event_index = event_index_offset + self.num_scheduled_events[queue_index]
                    ptr[queue_index, event_index] = sort_indices
                    val[queue_index, event_index] = W[i, sort_indices]
                    Jp = hstack((J, True))
                    added_events = event_index_offset[Jp] + 1
                    self.num_scheduled_events[queue_index[Jp]] += added_events
#                    print 'num_scheduled_events after', self.num_scheduled_events
#                    print 'dvecrow/ms', dvecrow/ms
#                    print 'sort_indices', sort_indices
#                    print 'int_delay', int_delay
#                    print 'queue_index', queue_index
#                    print 'J', J
#                    print 'K', K
#                    print 'A', A
#                    print 'B', B
#                    print 'BJ', BJ
#                    print 'event_index_offset', event_index_offset
#                    print 'event_index', event_index
#                    print 'added_events', added_events
#                    print 'queue_index[Jp]', queue_index[Jp]
#                    print 'ptr', ptr
#                    print 'val', val
#            print 'Spike propagation:'
#            print ptr
#            print val

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
