from brian import *
###### Set up the standard CUBA example ######
N = 4000
eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P = NeuronGroup(N, eqs, threshold=-50*mV,reset=-60*mV)
P.v = -60*mV+10*mV*rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
Ce = Connection(Pe, P, 'ge', weight=1.62*mV, sparseness=0.02)
Ci = Connection(Pi, P, 'gi', weight=-9*mV, sparseness=0.02)
###### Real time plotting stuff ######
# First of all, we use standard Brian monitors to record values
M = SpikeMonitor(P)
trace = StateMonitor(P, 'v', record=0)
# The ion() command switches pylab's "interactive mode" on
ion()
# We set up the plot with the correct axes
subplot(211)
# Note that we have to store a copy of the objects (plot lines) whose data
# we will update in real time
rasterline, = plot([], [], '.') # plot points, hence the '.'
axis([0, 1, 0, N])
subplot(212)
traceline, = plot([], []) # plot lines, hence no '.'
axis([0, 1, -0.06, -0.05])
# This network operation updates the graphics every 10ms of simulated time
@network_operation(clock=EventClock(dt=10*ms))
def draw_gfx():
    # This returns two lists i, t of the neuron indices and spike times for
    # all the recorded spikes so far
    i, t = zip(*M.spikes)
    # Now we update the raster and trace plots with this new data
    rasterline.set_xdata(t)
    rasterline.set_ydata(i)
    traceline.set_xdata(trace.times)
    traceline.set_ydata(trace[0])
    # and finally tell pylab to redraw it
    draw()

run(1*second)
draw_gfx() # final draw to get the last bits of data in
ioff() # switch interactive mode off
show() # and wait for user to close the window before shutting down