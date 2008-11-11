'''
This script shows four ways of attempting a standard type of task but which
have very different memory usage properties. The task is simply this::

    for i in range(15):
        G = NeuronGroup(1000, 'V:1', reset=0, threshold=-1)
        M = SpikeMonitor(G)
        run(100*ms)

The group ``G`` produces 1000 spikes every time step because the threshold is
lower than the reset, and the monitor ``M`` records them all. Although this
example is trivial, this sort of loop is very standard in writing simulations
that test multiple sets of parameters, or that average over several runs. This
naive way of doing it will use more than twice as much RAM as needed, and there
are some other ways of writing the loop above that will use all the RAM in the
system before crashing. The function ``del_and_collect_as_you_go()`` below
shows the right way of doing it to minimise memory usage.

The problem is that after the first loop, the G and M variables are still live
and so cannot be deallocated. At the beginning of the second loop, the G variable
is replaced by a reference to a new NeuronGroup but M still points to G and so
G won't be deallocated. At the second line of the second loop, M will be replaced
so that now in principle the original G and M can be reallocated but for whatever
reason Python doesn't do that immediately, it will do it at some point during the
``run(100*ms)`` or possibly during the third loop.

See the code below for some other cases to watch out for and a resolution. 
'''

from brian import *
import gc # Python's garbage collection module
N = 15

def naive_way():
    '''
    This way of doing it leaves the memory management entirely
    up to Python. On my system it peaks at around 250MB RAM used.
    '''
    for i in range(N):
        G = NeuronGroup(1000, 'V:1', reset=0, threshold=-1)
        M = SpikeMonitor(G)
        run(100*ms)
        print i+1, '/', N

def argh_big_problems():
    '''
    This way is even worse because there is now a circular reference
    so Python has to wait for a full garbage collection before it
    can deallocate memory. This version seemed to be growing without
    bounds, over 600MB before the end of the 4th loop and pretty much
    stalled at that point because it was using virtual memory. 
    '''
    for i in range(N):
        G = NeuronGroup(1000, 'V:1', reset=0, threshold=-1)
        G.M = SpikeMonitor(G)
        run(100*ms)
        print i+1, '/', N

def collect_as_you_go():
    '''
    We add in an explicit call for garbage collection which essentially
    doesn't improve it at all. This version still takes 250MB. If you
    use G.M instead of M it still goes nuts.
    '''
    for i in range(N):
        G = NeuronGroup(1000, 'V:1', reset=0, threshold=-1)
        M = SpikeMonitor(G)
        run(100*ms)
        gc.collect()
        print i+1, '/', N

def del_and_collect_as_you_go():
    '''
    Yay! A solution! This version has a peak memory usage of about 100MB
    and also runs significantly faster than the previous versions. It even
    works with the G.M circular reference.
    '''
    for i in range(N):
        G = NeuronGroup(1000, 'V:1', reset=0, threshold=-1)
        G.M = SpikeMonitor(G)
        run(100*ms)
        del G.M
        del G
        gc.collect()
        print i+1, '/', N

#naive_way()
#argh_big_problems()
#collect_as_you_go()
del_and_collect_as_you_go()