from brian import *

N = 10

burst_clock = EventClock(dt=100 * ms)

def burst_generator():
    start = burst_clock.t
    times = [(i, start + rand()*10 * ms) for i in range(N)]
    times.sort(key=lambda (i, t):t)
    return times

@network_operation(clock=burst_clock)
def next_burst():
    G.reinit()

G = SpikeGeneratorGroup(N, burst_generator)

M = SpikeMonitor(G)
run(500 * ms)
raster_plot(M)
show()
