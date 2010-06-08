from brian import *
from nose.tools import *
from brian.utils.approximatecomparisons import is_approx_equal

def test():
    '''
    :class:`Connection`
    ~~~~~~~~~~~~~~~~~~~
    
    **Initialised as:** ::
    
        Connection(source, target[, state=0[, delay=0*ms]])
    
    With arguments:
    
    ``source``
        The group from which spikes will be propagated.
    ``target``
        The group to which spikes will be propagated.
    ``state``
        The state variable name or number that spikes will be
        propagated to in the target group.
    ``delay``
        The delay between a spike being generated at the source
        and received at the target. At the moment, the mechanism
        for delays only works for relatively short delays (an
        error will be generated for delays that are too long), but
        this is subject to change. The exact behaviour then is
        not part of the assured interface, although it is very
        likely that the syntax will not change (or will at least
        be backwards compatible).
    
    **Methods**
    
    ``connect_random(P,Q,p[,weight=1])``
        Connects each neuron in ``P`` to each neuron in ``Q``.
    ``connect_full(P,Q[,weight=1])``
        Connect every neuron in ``P`` to every neuron in ``Q``.
    ``connect_one_to_one(P,Q)``
        If ``P`` and ``Q`` have the same number of neurons then neuron ``i``
        in ``P`` will be connected to neuron ``i`` in ``Q`` with weight 1.
    
    Additionally, you can directly access the matrix of weights by writing::
    
        C = Connection(P,Q)
        print C[i,j]
        C[i,j] = ...
    
    Where here ``i`` is the source neuron and ``j`` is the target neuron.
    Note: No unit checking is currently done if you use this method,
    but this is subject to change for future releases.

    The behaviour when a list of neuron ``spikes`` is received is to
    add ``W[i,:]`` to the target state variable for each ``i`` in ``spikes``. 
    '''
    reinit_default_clock()

    # test Connection object

    eqs = '''
    da/dt = 0.*hertz : 1.
    db/dt = 0.*hertz : 1.
    '''

    spikes = [(0, 1 * msecond), (1, 3 * msecond)]

    G1 = SpikeGeneratorGroup(2, spikes)
    G2 = NeuronGroup(2, model=eqs, threshold=10., reset=0.)

    # first test the methods
    # connect_full
    C = Connection(G1, G2)
    C.connect_full(G1, G2, weight=2.)
    for i in range(2):
        for j in range(2):
            assert (is_approx_equal(C[i, j], 2.))
    # connect_random
    C = Connection(G1, G2)
    C.connect_random(G1, G2, 0.5, weight=2.)
    # can't assert anything about that
    # connect_one_to_one
    C = Connection(G1, G2)
    C.connect_one_to_one(G1, G2)
    for i in range(2):
        for j in range(2):
            if i == j:
                assert (is_approx_equal(C[i, j], 1.))
            else:
                assert (is_approx_equal(C[i, j], 0.))
    del C
    # and we will use a specific set of connections in the next part
    Ca = Connection(G1, G2, 'a')
    Cb = Connection(G1, G2, 'b')
    Ca[0, 0] = 1.
    Ca[0, 1] = 1.
    Ca[1, 0] = 1.
    #Ca[1,1]=0 by default
    #Cb[0,0]=0 by default
    Cb[0, 1] = 1.
    Cb[1, 0] = 1.
    Cb[1, 1] = 1.
    net = Network(G1, G2, Ca, Cb)
    net.run(2 * msecond)
    # after 2 ms, neuron 0 will have fired, so a 0 and 1 should
    # have increased by 1 to [1,1], and b 1 should have increased
    # by 1 to 1
    assert (is_approx_equal(G2.a[0], 1.))
    assert (is_approx_equal(G2.a[1], 1.))
    assert (is_approx_equal(G2.b[0], 0.))
    assert (is_approx_equal(G2.b[1], 1.))
    net.run(2 * msecond)
    # after 4 ms, neuron 1 will have fired, so a 0 should have
    # increased by 1 to 2, and b 0 and 1 should have increased
    # by 1 to [1, 2]
    assert (is_approx_equal(G2.a[0], 2.))
    assert (is_approx_equal(G2.a[1], 1.))
    assert (is_approx_equal(G2.b[0], 1.))
    assert (is_approx_equal(G2.b[1], 2.))

    reinit_default_clock()

def test_poissoninputs():
    eqs = Equations("dv/dt=(1-v)/(1*second) : 1")
    group = NeuronGroup(N=1, model=eqs, reset=0, threshold=1)
    input = PoissonInputs(group, [], (10, 50 * Hz, .11, 'v'))
    m = SpikeCounter(group)
    sm = StateMonitor(group, 'v', record=True)
    net = Network(group, input, m, sm)
    net.run(500 * ms)
    assert (m.nspikes >= 1)

if __name__ == '__main__':
    test()
    test_poissoninputs()
