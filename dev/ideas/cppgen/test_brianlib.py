'''
Current progress:

+ LinearStateUpdater, does everything
    - not tested with multiple state variables yet, and this might be
      wrong (might be using the transpose of the matrix)
+ Threshold, working
+ Reset, working
+ CircularVector (only for dtype=int), working
+ SpikeContainer, working
+ NeuronGroup, only some bits implemented
    - so far a NeuronGroup has state, state updater, threshold and reset,
      the latter three all optional. It also has a SpikeContainer.
    - NeuronGroup has update and reset functions which work.
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
+ Connection
+ (everything else in Brian)

Notes:

+ With refractoriness, C++ is actually slower than Brian for N=10000,
  I think various factors are at play here.
    - My C++ code is written without thought to optimisation at the
      moment, in particular there are lots of pointer dereferences.
    - The C++ code returns spikes as a list<int> which is even passed
      by value rather than by reference, whereas the Brian code uses
      a static array to pass these. This is probably the most
      important point because it was adding the Refractoriness that
      caused the slowdown.

Ideas:

+ An interesting alternative to generating SpikeLists as in Brian is to
  just return a structure that points directly to the underlying
  CircularVector array. As long as the rest of the C++ code is aware
  that it is just getting a pointer to a CircularVector array you can
  thereby avoid a hell of a lot of copy operations.
'''
from brian import *
import brianlib as bl
import time
duration = 1*second
N = 10000
doplot = False
domonitor = False
debugmode = False
if debugmode:
    log_level_info()
######### Define a network we want to simulate ###############
eqs = '''
dV/dt = -(V-11*mV)/(10*ms) : volt
dW/dt = -(W-5*mV)/(50*ms) : volt
'''
G = NeuronGroup(N, eqs, threshold=10*mV, reset=0*mV, refractory=5*ms)
G.V = rand(N)*10*mV
G.W = rand(N)*5*mV
if domonitor:
    M = StateMonitor(G, 'V', record=True)
    M2 = StateMonitor(G, 'W', record=True)
net = Network()
net.add(G)
if domonitor:
    net.add(M)
    net.add(M2)
######### Convert to C++ versions from brianlib ##############
# This code is lengthy, but is obviously easily automatable (a user
# would certainly never have to do anything like this).
c = array(G._state_updater._C.flatten()) # if we don't do this the memory is corrupted
if debugmode:
    print 'c.flags', c.flags
    print 'A.flags', G._state_updater.A.flags
blGsu = bl.LinearStateUpdater(G._state_updater.A, c)
blGthr = bl.Threshold(G._threshold.state, float(G._threshold.threshold))
#blGreset = bl.Reset(G._resetfun.state, float(G._resetfun.resetvalue))
period = int(G._resetfun.period/G.clock.dt)+1
blGreset = bl.Refractoriness(G._resetfun.state, float(G._resetfun.resetvalue), period)
blG = bl.NeuronGroup(G._S, blGsu, blGthr, blGreset, G.LS.S.n, G.LS.ind.n)
if debugmode: print "brianlib.NeuronGroup instantiated OK:", G.LS.S.n, G.LS.ind.n
if domonitor:
    blM = bl.StateMonitor(blG, 0)
    blM2 = bl.StateMonitor(blG, 1)
blnet = bl.Network()
blnet.add(blG)
if domonitor:
    blnet.add(blM)
    blnet.add(blM2)
if debugmode:
    print "brianlib.Network instantiated OK"
    blG.update()
    print 'test update completed'
    blG.reset()
    print 'test reset completed'
########## Run the network in C++ and Brian ##################
# make a copy of V so we can run it twice
V = copy(G.V)
W = copy(G.W)
# Run in C++
start = time.time()
blnet.run(int(duration/defaultclock.dt))
end = time.time()
print 'C++ time:', (end-start)*second
if doplot and domonitor:
    subplot(221)
    for i in range(10):
        plot(blM[i])
    title('V, C++')
    subplot(222)
    for i in range(10):
        plot(blM2[i])
    title('W, C++')
# Run in Brian
G.V = V
G.W = W
start = time.time()
net.run(duration)
end = time.time()
print 'Brian time:', (end-start)*second
if doplot and domonitor:
    subplot(223)
    for i in range(10):
        plot(M[i])
    title('V, Brian')
    subplot(224)
    for i in range(10):
        plot(M2[i])
    title('W, Brian')
    show()