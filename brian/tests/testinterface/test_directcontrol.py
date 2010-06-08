from brian import *
from brian.utils.approximatecomparisons import *
from nose.tools import *

def test():
    """
    Spike containers
    ~~~~~~~~~~~~~~~~
    
    A spike container is either an iterable object or a callable object which
    returns an iterable object. For example, a list is a spike container, as
    is a generator, as is a function which returns a list, or a generator
    function.
    
    :class:`MultipleSpikeGeneratorGroup`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Called as::
    
        MultipleSpikeGeneratorGroup(spiketimes[,clock])
        
    spiketimes is a list of spike containers, one for each neuron in the group.
    The elements of the spike containers are spike times. Callable spike
    containers are called when the group is reinitialised. If you provide
    a generator rather than a callable object, reinitialising the group will
    not reinitialise the generator. If the containers are numpy arrays units
    will not be checked (times should be in seconds).
    
    So for example, the following will correspond to a group of 2 neurons, where
    the first fires at times 0ms, 2ms and 5ms, and the second fires at times
    1ms and 3ms::
    
        spiketimes = [[0*msecond, 2*msecond, 5*msecond],
                      [1*msecond, 3*msecond]]
        G = MultipleSpikeGeneratorGroup(spiketimes)
    
    You could do the same thing with generator functions (rather perversely in
    this case)::
    
        def st1():
            yield 0*msecond
            yield 2*msecond
            yield 5*msecond
        def st2():
            yield 1*msecond
            yield 3*msecond
        G = MultipleSpikeGeneratorGroup([st1(), st2()])

    Note that if two or more spike times fall within the same dt, spikes will stack up
    and come out one per dt until the stack is exhausted. A warning will be generated
    if this happens.
    
    If a clock is provided, updates of the group will be synchronised with
    that clock, otherwise the standard clock guessing procedure will be used
    (see :func:`~brian.clock.guess_clock` in the :mod:`~brian.clock` module).
    
    :class:`SpikeGeneratorGroup`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Called as::
    
        SpikeGeneratorGroup(N,spiketimes[,clock])
    
    where N is the number of neurons in the group, and spiketimes is a spike
    container, whose elements are tuples (i,t) meaning neuron i fires at
    time t. Pairs (i,t) need to be sorted in time unless spiketimes is a tuple
    or list. For example::
    
        from math import random
        def spikefirer(N,lower,upper):
            nexttime = random.uniform(lower,upper)
            while True:
                yield (random.randint(0,N-1),nexttime)
                nexttime = nexttime + random.uniform(lower,upper)
        G = SpikeGeneratorGroup(10,uniform_isi(10,0*msecond,10*msecond))
    
    would give a neuron group P with 10 neurons, where a random one of the neurons fires 
    with an interval between spikes which is uniform in (0ms, 10ms). If spiketimes is
    callable, it will be called when the group is reinitialised. If you provide
    a generator rather than a callable object, reinitialising the group will
    not reinitialise the generator.
    
    Note that if a neuron fires more than one spike in a given interval dt, additional
    spikes will be discarded.
    
    If a clock is provided, updates of the group will be synchronised with
    that clock, otherwise the standard clock guessing procedure will be used
    (see :func:`~brian.clock.guess_clock` in the :mod:`~brian.clock` module).
    
    :class:`PulsePacket`
    ~~~~~~~~~~~~~~~~~~~~
    
    Fires a Gaussian distributed packet of n spikes with given spread, called as::
    
        PulsePacket(t,n,sigma[,clock])
        
    You can change the parameters by calling the method ``generate(t,n,sigma)``.
    
    If a clock is provided, updates of the group will be synchronised with
    that clock, otherwise the standard clock guessing procedure will be used
    (see :func:`~brian.clock.guess_clock` in the :mod:`~brian.clock` module).
    """
    reinit_default_clock()

    # helper function for running a simple network
    def mininet(grouptype, *args, **kwds):
        reinit_default_clock()
        G = grouptype(*args, **kwds)
        M = SpikeMonitor(G, True)
        net = Network(G, M)
        net.run(5 * msecond)
        reinit_default_clock()
        return M.spikes

    # run mininet twice to see if it handles reinit() correctly
    def mininet2(grouptype, *args, **kwds):
        reinit_default_clock()
        G = grouptype(*args, **kwds)
        M = SpikeMonitor(G, True)
        net = Network(G, M)
        net.run(5 * msecond)
        spikes1 = M.spikes
        net.reinit()
        net.run(5 * msecond)
        spikes2 = M.spikes
        reinit_default_clock()
        return (spikes1, spikes2)

    # check multiple spike generator group with lists
    spiketimes = [[0 * msecond, 2 * msecond, 4 * msecond],
                  [1 * msecond, 3 * msecond]]
    spikes = mininet(MultipleSpikeGeneratorGroup, spiketimes)

    def test1(spikes):
        assert len(spikes) == 5
        i, t = zip(*spikes) # zip(*...) is the inverse of zip, so i is the ordered list of neurons that fired, and t is the ordered list of times
        assert i == (0, 1, 0, 1, 0) # check that the order of neuron firings is correct
        for s1, s2 in enumerate(t):
            assert is_approx_equal(s1 * msecond, s2) # the firing times are (0,1,2,3,4)ms
    test1(spikes)

    # check that given a different clock it works as expected, wrap in a function to stop magic functions from
    # picking up the clock objects we define here
    def testwithclock():
        spikes = mininet(MultipleSpikeGeneratorGroup, spiketimes, clock=Clock(dt=0.1 * msecond))
        test1(spikes)
        spikes = mininet(MultipleSpikeGeneratorGroup, spiketimes, clock=Clock(dt=2 * msecond))
        assert len(spikes) == 5
        i, t = zip(*spikes) # zip(*...) is the inverse of zip, so i is the ordered list of neurons that fired, and t is the ordered list of times
        for s1, s2 in zip([0, 2, 2, 4, 4], t):
            assert is_approx_equal(s1 * msecond, s2) # the firing times are (0,2,2,4,4)ms
    testwithclock()

    # check multiple spike generator group with generators
    def st1():
        yield 0 * msecond
        yield 2 * msecond
        yield 4 * msecond

    def st2():
        yield 1 * msecond
        yield 3 * msecond
    spikes = mininet(MultipleSpikeGeneratorGroup, [st1(), st2()])
    test1(spikes)

    # check reinit
    spikes1, spikes2 = mininet2(MultipleSpikeGeneratorGroup, [st1, st2])
    test1(spikes1)
    test1(spikes2)

    # spike generator with list
    spiketimes = [(0, 0 * msecond), (1, 1 * msecond), (0, 2 * msecond), (1, 3 * msecond), (0, 4 * msecond) ]
    spikes = mininet(SpikeGeneratorGroup, 2, spiketimes)
    test1(spikes)

    # check that it works with a clock
    def testwithclock():
        spikes = mininet(SpikeGeneratorGroup, 2, spiketimes, clock=Clock(dt=0.1 * msecond))
        test1(spikes)
        spikes = mininet(SpikeGeneratorGroup, 2, spiketimes, clock=Clock(dt=2 * msecond))
        assert len(spikes) == 5
        i, t = zip(*spikes) # zip(*...) is the inverse of zip, so i is the ordered list of neurons that fired, and t is the ordered list of times
        for s1, s2 in zip([0, 2, 2, 4, 4], t):
            assert is_approx_equal(s1 * msecond, s2) # the firing times are (0,2,2,4,4)ms
    testwithclock()

    # spike generator with generator
    def sg():
        yield (0, 0 * msecond)
        yield (1, 1 * msecond)
        yield (0, 2 * msecond)
        yield (1, 3 * msecond)
        yield (0, 4 * msecond)
    spikes = mininet(SpikeGeneratorGroup, 2, sg())
    test1(spikes)

    # spike generator reinit
    spikes1, spikes2 = mininet2(SpikeGeneratorGroup, 2, sg)
    test1(spikes1)
    test1(spikes2)

    # pulse packet with 0 spread
    spikes = mininet(PulsePacket, 2.5 * msecond, 10, 0 * msecond)
    assert len(spikes) == 10
    i, t = zip(*spikes)
    for s in t:
        assert is_approx_equal(2.5 * msecond, s)
    # do not attempt to verify the behaviour of PulsePacket here, this is
    # an interface test only

if __name__ == '__main__':
    test()
