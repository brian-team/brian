'''
Tutorial 1e: Connecting neurons
*******************************

In the previous parts of this tutorial, the neurons are
still all unconnected. We add in connections here. The
model we use is that when neuron i is connected to 
neuron j and neuron i fires a spike, then the membrane
potential of neuron j is instantaneously increased by
a value ``psp``. We start as before:
'''
from brian import *

tau = 20 * msecond        # membrane time constant
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
El = -49 * mvolt          # resting potential (same as the reset)
'''
Now we include a new parameter, the PSP size:
'''
psp = 0.5 * mvolt         # postsynaptic potential size
'''
And continue as before:
'''
G = NeuronGroup(N=40, model='dV/dt = -(V-El)/tau : volt',
              threshold=Vt, reset=Vr)
'''
Connections
~~~~~~~~~~~

We now proceed to connect these neurons. Firstly, we declare
that there is a connection from neurons in ``G`` to neurons in ``G``.
For the moment, this is just something that is necessary to
do, the reason for doing it this way will become clear in the
next tutorial.
'''
C = Connection(G, G)
'''
Now the interesting part, we make these neurons be randomly
connected with probability 0.1 and weight ``psp``. Each neuron
i in ``G`` will be connected to each neuron j in ``G`` 
with probability 0.1. The weight of the connection is the
amount that is added to the membrane potential of the target
neuron when the source neuron fires a spike.
'''
C.connect_random(sparseness=0.1, weight=psp)
'''
These two previous lines could be done in one line::

  C = Connection(G,G,sparseness=0.1,weight=psp)

Now we continue as before:
'''
M = SpikeMonitor(G)

G.V = Vr + rand(40) * (Vt - Vr)

run(1 * second)

print M.nspikes
'''
You can see that the number of spikes has jumped from around
800-850 to around 1000-1200. In the next part of the tutorial,
we'll look at a way to plot the output of the network.

Exercise
~~~~~~~~

Try varying the parameter ``psp`` and see what happens. How large
can you make the number of spikes output by the network? Why?

Solution
~~~~~~~~

The logically maximum number of firings is
400,000 = 40 * 1000 / 0.1, the number of neurons in the
network * the time it runs for / the integration step size (you
cannot have more than one spike per step).

In fact, the number of firings is bounded above by 200,000. The
reason for this is that the network updates in the following way:

1. Integration step
2. Find neurons above threshold
3. Propagate spikes
4. Reset neurons which spiked

You can see then that if neuron i has spiked at time t, then it
will not spike at time t+dt, even if it receives spikes from
another neuron. Those spikes it receives will be added at step
3 at time t, then reset to ``Vr`` at step 4 of time t, then the
thresholding function at time t+dt is applied at step 2, before
it has received any subsequent inputs. So the most a neuron
can spike is every other time step.
'''
