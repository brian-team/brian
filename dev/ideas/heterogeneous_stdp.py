from brian import *

clk = Clock()

interval = 50*ms
repeats = 20
N = 10
max_delay = 2.5*ms
offset = 1.25*ms
dA = 0.1
stdp_tau = 10*ms

G = SpikeGeneratorGroup(N, [(i, t*second) for i in xrange(N) for t in arange(repeats)*interval], clock=clk)
Gd = SpikeGeneratorGroup(N, [(i, t*second+i*max_delay/N) for i in xrange(N) for t in arange(repeats)*interval], clock=clk)
Hi = SpikeGeneratorGroup(1, [(0, t*second+offset-defaultclock.dt) for t in arange(repeats)*interval], clock=clk)
H = NeuronGroup(1, 'V:1', reset=0, threshold=3*N, clock=clk)
Hd = NeuronGroup(1, 'V:1', reset=0, threshold=3*N, clock=clk)
CH = Connection(Hi, H, 'V', weight=3*N+1)
CHd = Connection(Hi, Hd, 'V', weight=3*N+1)
C = DelayConnection(G, H, weight=1, max_delay=max_delay, structure='sparse')
for i in arange(N):
    C.delay[i,0] = i*max_delay/N
Cd = Connection(Gd, Hd, weight=1, structure='dense')

stdpd = ExponentialSTDP(Cd, stdp_tau, stdp_tau, dA, -dA, wmax=2.)
stdp = ExponentialSTDP(C, stdp_tau, stdp_tau, dA, -dA, wmax=2.)

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
run(repeats*interval)
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