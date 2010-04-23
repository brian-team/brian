from brian import *
import sys

log_level_warn()

if 0:
    class Clock(Clock):
        def tick(self):
            print 'Ticking:', id(self), self
            super(Clock, self).tick()

def repfunc(layer1):
    def rep(spikes):
        print 'rep:', id(layer1), layer1.clock.t, spikes
    return rep

for ii in range(0,5):
    # Setup a new clock
    sim_clk=Clock(dt=1*ms)
    if 0:
        print 'This sim clock:', id(sim_clk)

    # Setup up a particular sequence of spikes
    spiketimes = [(0,1*ms),(0,2*ms),(0,3*ms),(0,4*ms)]
    layer1 = SpikeGeneratorGroup(2,spiketimes,clock=sim_clk)
    if 0:
        print 'SGG clock:', id(layer1.clock)

    # Setup output recorder
    spikes1=SpikeMonitor(layer1)
    if 0:
        spikes2 = SpikeMonitor(layer1, function=repfunc(layer1))

    if 0:
        @network_operation(clock=sim_clk)
        def rep2():
            print 'NAS:', layer1._next_allowed_spiketime

    net = MagicNetwork()
    net.prepare()
    if 1:
        print 'Groups:', map(id, net.groups)
        print 'Connections:', map(id, net.connections)
        print 'Update schedule:'
        for clk, v in net._update_schedule.items():
            print '  Clock:', id(clk)
            print '  Schedule:', v
        sys.stdout.flush()
    if 0:
        if hasattr(net, 'clocks'):
            print 'Clocks =', map(id, net.clocks)
    net.run(5*ms)

    print 'Counted', spikes1.nspikes, 'spikes.'
    
    #forget(layer1)
    #clear()
    