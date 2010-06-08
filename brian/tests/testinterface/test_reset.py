from brian import *
from nose.tools import *
from brian.utils.approximatecomparisons import is_approx_equal

def test():
    """
    :class:`Reset`
    ~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = Reset(resetvalue=0*mvolt, state=0)

    After a neuron from a group with this reset fires, it
    will set the specified state variable to the given value.
    State variable 0 is customarily the membrane voltage,
    but this isn't required. 
    
    :class:`FunReset`
    ~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = FunReset(resetfun)
    
    Where resetfun is a function taking two arguments, the group
    it is acting on, and the indices of the spikes to be reset.
    The following is an example reset function::
    
        def f(P,spikeindices):
            P._S[0,spikeindices]=array([i/10. for i in range(len(spikeindices))])    
    
    :class:`Refractoriness`
    ~~~~~~~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = Refractoriness(resetvalue=0*mvolt,period=5*msecond,state=0)
    
    After a neuron from a group with this reset fires, the specified state
    variable of the neuron will be set to the specified resetvalue for the
    specified period.
    
    :class:`NoReset`
    ~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = NoReset()
        
    Does nothing.
    """
    reinit_default_clock()


    # test that reset works as expected
    # the setup below is that group G starts with state values (1,1,1,1,1,0,0,0,0,0) threshold
    # value 0.5 (which should be initiated for the first 5 neurons) and reset 0.2 so that the
    # final state should be (0.2,0.2,0.2,0.2,0.2,0,0,0,0,0) 
    G = NeuronGroup(10, model=LazyStateUpdater(), reset=Reset(0.2), threshold=Threshold(0.5), init=(0.,))
    G1 = G.subgroup(5)
    G2 = G.subgroup(5)
    G1.state(0)[:] = array([1.] * 5)
    G2.state(0)[:] = array([0.] * 5)
    net = Network(G)
    net.run(1 * msecond)
    assert (all(G1.state(0) < 0.21) and all(0.19 < G1.state(0)) and all(G2.state(0) < 0.01))

    # check that function reset works as expected
    def f(P, spikeindices):
        P._S[0, spikeindices] = array([i / 10. for i in range(len(spikeindices))])
        P.called_f = True
    G = NeuronGroup(10, model=LazyStateUpdater(), reset=FunReset(f), threshold=Threshold(2.), init=(3.,))
    G.called_f = False
    net = Network(G)
    net.run(1 * msecond)
    assert (G.called_f)
    for i, v in enumerate(G.state(0)):
        assert (is_approx_equal(i / 10., v))

    # check that refractoriness works as expected
    # the network below should start at V=15, immediately spike as it is above threshold=1,
    # then should be clamped at V=-.5 until t=1ms at which point it should quickly evolve
    # via the DE to a value near 0 (and certainly between -.5 and 0). We test that the
    # value at t=0.5 is exactly -.5 and the value at t=1.5 is between -0.4 and 0.1 (to
    # avoid floating point problems)
    dV = 'dV/dt=-V/(.1*msecond):1.'
    G = NeuronGroup(1, model=dV, threshold=1., reset=Refractoriness(-.5, 1 * msecond))
    G.V = 15.
    net = Network(G)
    net.run(0.5 * msecond)
    for v in G.state(0):
        assert (is_approx_equal(v, -.5))
    net.run(1 * msecond)
    for v in G.state(0):
        assert (-0.4 < v < 0.1)

    get_default_clock().reinit()

if __name__ == '__main__':
    test()
