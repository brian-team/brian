#!/usr/bin/env python
'''
Realtime plotting example
'''

# These lines are necessary for interactive plotting when launching from the
# Eclipse IDE, they may not be necessary in every environment.
import matplotlib
matplotlib.use('WXAgg') # You may need to experiment, try WXAgg, GTKAgg, QTAgg, TkAgg

from brian import *
###### Set up the standard CUBA example ######
N = 4000
eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P = NeuronGroup(N, eqs, threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
Ce = Connection(Pe, P, 'ge', weight=1.62 * mV, sparseness=0.02)
Ci = Connection(Pi, P, 'gi', weight= -9 * mV, sparseness=0.02)

M = SpikeMonitor(P)
trace = RecentStateMonitor(P, 'v', record=range(5), duration=200 * ms)

ion()
subplot(211)
raster_plot(M, refresh=10 * ms, showlast=200 * ms, redraw=False)
subplot(212)
trace.plot(refresh=10 * ms, showlast=200 * ms)

run(1 * second)

ioff() # switch interactive mode off
show() # and wait for user to close the window before shutting down
