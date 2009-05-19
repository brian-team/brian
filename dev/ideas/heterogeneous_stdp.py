from brian import *
from time import time

clk = Clock()

do_monitor = True
do_normal_connection = True
do_delayed_connection = True
use_stdp = True
structure = 'sparse'

interval = 50*ms
repeats = 20
N = 10
M = 1
max_delay = 2.5*ms
offset = 1.25*ms
dA = 0.1
stdp_tau = 10*ms

if do_delayed_connection:
    G = SpikeGeneratorGroup(N, [(i, t*second) for i in xrange(N) for t in arange(repeats)*interval], clock=clk)
if do_normal_connection:
    Gd = SpikeGeneratorGroup(N, [(i, t*second+i*max_delay/N) for i in xrange(N) for t in arange(repeats)*interval], clock=clk)
Hi = SpikeGeneratorGroup(M, [(i, t*second+offset-defaultclock.dt) for t in arange(repeats)*interval for i in xrange(M)], clock=clk)
if do_delayed_connection:
    H = NeuronGroup(M, 'V:1', reset=0, threshold=3*N, clock=clk)
    CH = IdentityConnection(Hi, H, 'V', weight=3*N+1)
if do_normal_connection:
    Hd = NeuronGroup(M, 'V:1', reset=0, threshold=3*N, clock=clk)
    CHd = IdentityConnection(Hi, Hd, 'V', weight=3*N+1)
if do_delayed_connection:
    C = DelayConnection(G, H, weight=1, max_delay=max_delay, structure=structure)
    for j in xrange(M):
        C.delay[:,j] = arange(N)*max_delay/N
#    for i in arange(N):
#        for j in xrange(M):
#            C.delay[i,j] = i*max_delay/N
if do_normal_connection:
    Cd = Connection(Gd, Hd, weight=1, structure=structure)

if use_stdp:
    if do_normal_connection:
        stdpd = ExponentialSTDP(Cd, stdp_tau, stdp_tau, dA, -dA, wmax=2.)
    if do_delayed_connection:
        stdp = ExponentialSTDP(C, stdp_tau, stdp_tau, dA, -dA, wmax=2.)

if do_monitor:
    @network_operation(when='end', clock=Clock(dt=interval*repeats/20.))
    def update_weight_plots(clk):
        c = clk.t/(interval*repeats)
        c = (c, 0, 1-c)
        subplot(243)
        plot(Cd.W[:,0], color=c)
        subplot(247)
        plot(C.W[:,0], color=c)
    G.spikemon = SpikeMonitor(G)
    Gd.spikemon = SpikeMonitor(Gd)
    H.spikemon = SpikeMonitor(H)
    H.statemon = MultiStateMonitor(H, record=True, clock=clk)
    Hd.spikemon = SpikeMonitor(Hd)
    Hd.statemon = MultiStateMonitor(Hd, record=True, clock=clk)
    stdpmons = [MultiStateMonitor(g, vars=[0], record=True, clock=clk) for g in stdp.contained_objects if isinstance(g, NeuronGroup)]
    stdpdmons = [MultiStateMonitor(g, vars=[0], record=True, clock=clk) for g in stdpd.contained_objects if isinstance(g, NeuronGroup)]
run(0*ms)
start = time()
run(repeats*interval)
print time()-start
if do_monitor:
    subplot(241)
    raster_plot(Gd.spikemon)
    raster_plot(Hd.spikemon)
    subplot(245)
    raster_plot(G.spikemon)
    raster_plot(H.spikemon)
    subplot(242)
    Hd.statemon.plot()
    subplot(246)
    H.statemon.plot()
    subplot(244)
    for m in stdpdmons:
        m.plot()
    subplot(248)
    for m in stdpmons:
        m.plot()
    show()