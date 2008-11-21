'''
Current progress:

+ LinearStateUpdater, does everything
    - not tested with multiple state variables yet, and this might be
      wrong (might be using the transpose of the matrix)
+ Threshold, working
+ Reset, working
+ NeuronGroup, only some bits implemented
    - so far a NeuronGroup has state, state updater, threshold and reset,
      the latter three all optional.
    - NeuronGroup implements last spikes by keeping a copy of the spike
      list returned by the Threshold
    - NeuronGroup has an update function which works correctly, except
      that there is no circular array of spikes, only the last spikes
+ StateMonitor, very crude
    - Can only record a given state variable for all neurons at all
      time steps
    - Can only get the values via __getitem__(i) and this returns a
      list rather than an array
    - Is derived from an essentially empty NetworkOperation class
      which doesn't include the when option at the moment
+ Network
    - Can add and run multiple NeuronGroup and NetworkOperation objs.
    - No clocks used

Still missing:

+ Clocks
+ CircularVector and SpikeContainer
+ Connection
+ (everything else in Brian)
'''
from brian import *
import brianlib as bl
import time
duration = 10*second
N = 10000
doplot = False
domonitor = False
######### Define a network we want to simulate ###############
eqs = '''
dV/dt = -(V-11*mV)/(10*ms) : volt
'''
G = NeuronGroup(N, eqs, threshold=10*mV, reset=0*mV)
G.V = rand(N)*10*mV
if domonitor: M = StateMonitor(G, 'V', record=True)
net = Network()
net.add(G)
if domonitor: net.add(M)
######### Convert to C++ versions from brianlib ##############
# This code is lengthy, but is obviously easily automatable (a user
# would certainly never have to do anything like this).
c = array(G._state_updater._C.flatten()) # if we don't do this the memory is corrupted
blGsu = bl.LinearStateUpdater(G._state_updater.A, c)
blGthr = bl.Threshold(G._threshold.state, float(G._threshold.threshold))
blGreset = bl.Reset(G._resetfun.state, float(G._resetfun.resetvalue))
blG = bl.NeuronGroup(G._S, blGsu, blGthr, blGreset)
if domonitor: blM = bl.StateMonitor(blG, 0)
blnet = bl.Network()
blnet.add(blG)
if domonitor: blnet.add(blM)
########## Run the network in C++ and Brian ##################
# make a copy of V so we can run it twice
V = copy(G.V)
# Run in C++
start = time.time()
blnet.run(int(duration/defaultclock.dt))
end = time.time()
print 'C++ time:', (end-start)*second
if doplot and domonitor:
    subplot(121)
    for i in range(10):
        plot(blM[i])
    title('C++')
# Run in Brian
G.V = V
start = time.time()
net.run(duration)
end = time.time()
print 'Brian time:', (end-start)*second
if doplot and domonitor:
    subplot(122)
    for i in range(10):
        plot(M[i])
    title('Brian')
    show()