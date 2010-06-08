'''
Tutorial 1d: Introducing randomness
***********************************

In the previous part of the tutorial, all the neurons start
at the same values and proceed deterministically, so they all
spike at exactly the same times. In this part, we introduce
some randomness by initialising all the membrane potentials
to uniform random values between the reset and threshold
values.

We start as before:
'''
from brian import *

tau = 20 * msecond        # membrane time constant
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
El = -49 * mvolt          # resting potential (same as the reset)

G = NeuronGroup(N=40, model='dV/dt = -(V-El)/tau : volt',
              threshold=Vt, reset=Vr)

M = SpikeMonitor(G)
'''
But before we run the simulation, we set the values of the
membrane potentials directly. The notation ``G.V`` refers
to the array of values for the variable ``V`` in group ``G``. In
our case, this is an array of length 40. We set its values
by generating an array of random numbers using Brian's
``rand`` function. The syntax is ``rand(size)`` generates an
array of length ``size`` consisting of uniformly distributed
random numbers in the interval 0, 1.
'''
G.V = Vr + rand(40) * (Vt - Vr)
'''
And now we run the simulation as before.
'''
run(1 * second)

print M.nspikes
'''
But this time we get a varying number of spikes each time
we run it, roughly between 800 and 850 spikes. In the
next part of this tutorial, we introduce a bit more
interest into this network by connecting the neurons together.
'''
