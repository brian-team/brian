#!/usr/bin/env python
'''
Example of using derived classes in Brian

Using a class derived from one of Brian's classes can be a useful way of
organising code in complicated simulations. A class such as a :class:`NeuronGroup`
can itself create further :class:`NeuronGroup`, :class:`Connection` and
:class:`NetworkOperation` objects. In order to have these objects included in
the simulation, the derived class has to include them in its ``contained_objects``
list (this tells Brian to add these to the :class:`Network` when the derived
class object is added to the network).
'''
from brian import *


class PoissonDrivenGroup(NeuronGroup):
    '''
    This class is a group of leaky integrate-and-fire neurons driven by
    external Poisson inputs. The class creates the Poisson inputs and
    connects them to itself. 
    '''
    def __init__(self, N, rate, weight):
        tau = 10 * ms
        eqs = '''
        dV/dt = -V/tau : 1
        '''
        # It's essential to call the initialiser of the base class
        super(PoissonDrivenGroup, self).__init__(N, eqs, reset=0, threshold=1)
        self.poisson_group = PoissonGroup(N, rate)
        self.conn = Connection(self.poisson_group, self, 'V')
        self.conn.connect_one_to_one(weight=weight)
        self.contained_objects += [self.poisson_group,
                                   self.conn]

G = PoissonDrivenGroup(100, 100 * Hz, .3)

M = SpikeMonitor(G)
M_pg = SpikeMonitor(G.poisson_group)
trace = StateMonitor(G, 'V', record=0)

run(1 * second)

subplot(311)
raster_plot(M_pg)
title('Input spikes')
subplot(312)
raster_plot(M)
title('Output spikes')
subplot(313)
plot(trace.times, trace[0])
title('Sample trace')
show()
