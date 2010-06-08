'''
Tutorial 1b: Counting spikes
****************************

In the previous part of the tutorial we looked at the following:

* Importing the Brian module into Python
* Using quantities with units
* Defining a neuron model by its differential equation
* Creating a group of neurons
* Running a network

In this part, we move on to looking at the output of the network.
    
The first part of the code is the same.
'''
from brian import *

tau = 20 * msecond        # membrane time constant
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
El = -60 * mvolt          # resting potential (same as the reset)

G = NeuronGroup(N=40, model='dV/dt = -(V-El)/tau : volt',
              threshold=Vt, reset=Vr)
'''
Counting spikes
~~~~~~~~~~~~~~~

Now we would like to have some idea of what this network is
doing. In Brian, we use monitors to keep track of the behaviour
of the network during the simulation. The simplest monitor of
all is the :class:`SpikeMonitor`, which just records the spikes from a
given :class:`NeuronGroup`.
'''
M = SpikeMonitor(G)
'''
Results
~~~~~~~

Now we run the simulation as before:
'''
run(1 * second)
'''
And finally, we print out how many spikes there were:
'''
print M.nspikes
'''
So what's going on? Why are there 40 spikes? Well, the answer is
that the initial value of the membrane potential for every neuron
is 0 mV, which is above the threshold potential of -50 mV and so there
is an initial spike at t=0 and then it resets to -60 mV and stays there,
below the threshold potential. In the next part of this tutorial, we'll
make sure there are some more spikes to see.
'''
