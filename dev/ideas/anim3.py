from brian import *
import pylab

N = 3

G = NeuronGroup(N, 'dV/dt=-V/(10*ms)+xi/(10*ms)**.5:1', reset=0, threshold=1)
M = StateMonitor(G, 'V', record=True)

ion()

lines = [line for line, in [plot([0], [0]) for _ in range(N)]]
axis([0, .1, -1, 1])

@network_operation(clock=EventClock(dt=2*ms))
def update_plot():
    for i, line in enumerate(lines):
        line.set_xdata(M.times)
        line.set_ydata(M[i])
    draw()

run(.1*second)
ioff()
show()