#!/usr/bin/env python
'''
Network of noisy IF neurons with gap junctions
'''
from brian import *

N = 300
v0 = 5 * mV
tau = 20 * ms
sigma = 5 * mV
vt = 10 * mV
vr = 0 * mV
g_gap = 1. / N
beta = 60 * mV * 2 * ms
delta = vt - vr

eqs = '''
dv/dt=(v0-v)/tau+g_gap*(u-N*v)/tau : volt
du/dt=(N*v0-u)/tau : volt # input from other neurons
'''

def myreset(P, spikes):
    P.v[spikes] = vr # reset
    P.v += g_gap * beta * len(spikes) # spike effect
    P.u -= delta * len(spikes)

group = NeuronGroup(N, model=eqs, threshold=vt, reset=myreset)

@network_operation
def noise(cl):
    x = randn(N) * sigma * (cl.dt / tau) ** .5
    group.v += x
    group.u += sum(x)

trace = StateMonitor(group, 'v', record=[0, 1])
spikes = SpikeMonitor(group)
rate = PopulationRateMonitor(group)

run(1 * second)
subplot(311)
raster_plot(spikes)
subplot(312)
plot(trace.times / ms, trace[0] / mV)
plot(trace.times / ms, trace[1] / mV)
subplot(313)
plot(rate.times / ms, rate.smooth_rate(5 * ms) / Hz)
show()
