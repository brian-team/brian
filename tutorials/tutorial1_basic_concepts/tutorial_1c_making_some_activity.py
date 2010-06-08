'''
Tutorial 1c: Making some activity
*********************************

In the previous part of the tutorial we found that each neuron
was producing only one spike. In this part, we alter the model so
that some more spikes will be generated. What we'll do is alter
the resting potential ``El`` so that it is above threshold, this
will ensure that some spikes are generated. The first few
lines remain the same: 
'''
from brian import *

tau = 20 * msecond        # membrane time constant
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
'''
But we change the resting potential to -49 mV, just above the
spike threshold:
'''
El = -49 * mvolt          # resting potential (same as the reset)
'''
And then continue as before:
'''
G = NeuronGroup(N=40, model='dV/dt = -(V-El)/tau : volt',
              threshold=Vt, reset=Vr)

M = SpikeMonitor(G)

run(1 * second)

print M.nspikes
'''
Running this program gives the output ``840``. That's because
every neuron starts at the same initial value and proceeds
deterministically, so that each neuron fires at exactly the
same time, in total 21 times during the 1s of the run.

In the next part, we'll introduce a random element into the
behaviour of the network.

Exercises
~~~~~~~~~

1. Try varying the parameters and seeing how the number of
   spikes generated varies.
2. Solve the differential equation by hand and compute a
   formula for the number of spikes generated. Compare this
   with the program output and thereby partially verify it.
   (Hint: each neuron starts at above the threshold and so
   fires a spike immediately.)

Solution
~~~~~~~~

Solving the differential equation gives:

    V = El + (Vr-El) exp (-t/tau)

Setting V=Vt at time t gives:

    t = tau log( (Vr-El) / (Vt-El) )

If the simulator runs for time T, and fires a spike immediately
at the beginning of the run it will then generate n spikes,
where:

    n = [T/t] + 1

If you have m neurons all doing the same thing, you get nm
spikes. This calculation with the parameters above gives:

    t = 48.0 ms
    n = 21
    nm = 840

As predicted.
'''
