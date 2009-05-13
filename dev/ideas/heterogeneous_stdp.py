from brian import *

interval = 10*ms
repeats = 20
N = 10
max_delay = 5*ms
offset = 2.5*ms

G = SpikeGeneratorGroup(N, [(i, t*second) for i in xrange(10) for t in arange(repeats)*interval])
Gd = SpikeGeneratorGroup(N, [(i, t*second+i*max_delay/N) for i in xrange(10) for t in arange(repeats)*interval])
Hi = SpikeGeneratorGroup(1, [(0, t*second+offset-defaultclock.dt) for t in arange(repeats)*interval])
H = NeuronGroup(1, 'V:1', reset=0, threshold=3*N)
Hd = NeuronGroup(1, 'V:1', reset=0, threshold=3*N)
CH = Connection(Hi, H, 'V', weight=3*N+1)
CHd = Connection(Hi, Hd, 'V', weight=3*N+1)
C = DelayConnection(G, H, weight=1, max_delay=max_delay, structure='dense')
C.delay[:, 0] = linspace(0*second, max_delay, N)
Cd = Connection(Gd, Hd, weight=1, structure='dense')

stdp = ExponentialSTDP(C, 10*ms, 10*ms, 1.0, -1.0, wmax=2.)
stdpd = ExponentialSTDP(Cd, 10*ms, 10*ms, 1.0, -1.0, wmax=2.)

subplot(243)
plot(Cd.W[:,0])
title('W start')
subplot(247)
plot(C.W[:,0])
title('W start delays')
G.spikemon = SpikeMonitor(G)
Gd.spikemon = SpikeMonitor(Gd)
H.spikemon = SpikeMonitor(H)
H.statemon = MultiStateMonitor(H, record=True)
Hd.spikemon = SpikeMonitor(Hd)
Hd.statemon = MultiStateMonitor(Hd, record=True)
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
plot(Cd.W[:,0])
title('W end')
subplot(248)
plot(C.W[:,0])
title('W end delays')
show()