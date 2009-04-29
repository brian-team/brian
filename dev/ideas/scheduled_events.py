'''
Simple implementation of an event scheduling scheme for Brian.

The idea is that a spike schedules a future event *p += val for p a pointer
and val a float value. Below this idea is used to implement dendritic delays
in an alternative way to DelayConnection. You could implement axonal delays by
using *p += *q for p and q both pointers (here q would be a pointer to an
appropriate entry in a weight matrix, and p an appropriate neuron state variable).
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
                ptr, val = self.scheduled_events_ptr, self.scheduled_events_val
                index = self.scheduled_events_index
                print 'Event processing:'
                print ptr[index, :]
                print val[index, :]
                code = '''
                for(int i=0; i<num_scheduled_events; i++)
                {
                    double *p = (double *)(ptr(index, i));
                    *p += val(index, i);
                }
                '''
                weave.inline(code, ['num_scheduled_events', 'ptr', 'val', 'index'],
                             type_converters=weave.converters.blitz,
                             compiler='gcc')
                # clean up and move on
                self.num_scheduled_events[self.scheduled_events_index] = 0
            self.scheduled_events_index = (self.scheduled_events_index+1)%self._max_delay 
        self.contained_objects.append(process_scheduled_events)
    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
#            DelayConnection.propagate(self, spikes)
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
                    int offset = (index+(int)(idt*dvec(k, j)))%md;
                    int e = numevents(offset)++;
                    ptr(offset, e) = (int)(&(sv(j)));
                    val(offset, e) = W(k, j);
                }
            }
            '''
            weave.inline(code, ['numevents', 'ptr', 'val', 'nspikes', 'n', 'm', 'index', 'idt',
                                'md', 'dvec', 'sv', 'spikes', 'W'],
                         type_converters=weave.converters.blitz,
                         compiler='gcc')
            print 'Spike propagation:'
            print ptr
            print val

C = ScheduledEventDelayConnection(G, H, structure='dense', max_delay=1*ms)
C.connect_full(G, H, weight=1, delay=(0*ms, 1*ms))

G.spikemon = SpikeMonitor(G)
H.spikemon = SpikeMonitor(H)
G.statemon = MultiStateMonitor(G, record=True)
H.statemon = MultiStateMonitor(H, record=True)

G.V = 1

run(1*ms)

subplot(221)
raster_plot(G.spikemon)
subplot(222)
raster_plot(H.spikemon)
subplot(223)
G.statemon.plot()
subplot(224)
H.statemon.plot()
show()