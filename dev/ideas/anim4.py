'''
Very short example program.
'''
from brian import *

N=4000        # number of neurons
Ne=int(N*0.8) # excitatory neurons 
Ni=N-Ne       # inhibitory neurons
p=80./N
duration=1000*ms

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P=NeuronGroup(N,model=eqs,
              threshold=-50*mV,reset=-60*mV)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)

Ce=Connection(Pe,P,'ge',weight=1.62*mV,sparseness=p)
Ci=Connection(Pi,P,'gi',weight=-9*mV,sparseness=p)

M = SpikeMonitor(P)
trace = StateMonitor(P, 'v', record=0)

ion()
subplot(211)
rasterline, = plot([], [], '.')
axis([0, 1, 0, N])
subplot(212)
traceline, = plot([], [])
axis([0, 1, -0.06, -0.05])

@network_operation(clock=EventClock(dt=10*ms))
def draw_gfx():
    i, t = zip(*M.spikes)
    rasterline.set_xdata(t)
    rasterline.set_ydata(i)
    traceline.set_xdata(trace.times)
    traceline.set_ydata(trace[0])
    draw()

run(1*second)
draw_gfx()
ioff()
show()