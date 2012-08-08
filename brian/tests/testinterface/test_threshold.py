from brian import *
from nose.tools import *
from operator import itemgetter
from brian.utils.approximatecomparisons import is_approx_equal
from brian.tests import repeat_with_global_opts
from brian.globalprefs import get_global_preference

@repeat_with_global_opts([
                          # no C code or code generation,
                          {'useweave': False, 'usecodegen': False},
                          # # use weave but no code generation 
                          {'useweave': True, 'usecodegen': False}, 
                          # use Python code generation
                          {'useweave': False, 'usecodegen': True,
                           'usecodegenthreshold': True},
                          # use C code generation
                          {'useweave': True, 'usecodegen': True,
                           'usecodegenthreshold': True, 'usecodegenweave': True}
                          ])
def test():
    """
    :class:`Threshold`
    ~~~~~~~~~~~~~~~~~~
    
    Initialised as ``Threshold(threshold[,state=0])``
    
    Causes a spike whenever the given state variable is above
    the threshold value.
    
    :class:`NoThreshold`
    ~~~~~~~~~~~~~~~~~~~~
    
    Does nothing, initialised as ``NoThreshold()``
    
    Functional thresholds
    ~~~~~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        FunThreshold(thresholdfun)
        SimpleFunThreshold(thresholdfun[,state=0])
    
    Threshold functions return a boolean array the same size as the
    number of neurons in the group, where if the returned array is
    True at index i then neuron i fires.
    
    The arguments passed to the :class:`FunThreshold` function are the
    full array of state variables for the group in order.
    
    The argument passed to the :class:`SimpleFunThreshold` function is the
    array of length N corresponding to the given state variable.
    
    :class:`VariableThreshold`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Initialised as ``VariableThreshold(threshold_state[,state=0])``
    
    Causes a spike whenever the state variable defined by state
    is above the state variable defined by threshold_state.
    
    :class:`EmpiricalThreshold`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        EmpiricalThreshold(threshold[,refractory=1*msecond[,state=0[,clock]]])
    
    Causes a spike when the given state variable exceeds the threshold value,
    but only if there has been no spike within the refractory period. Will
    use the given clock if specified, otherwise the standard guessing procedure
    is used.
    
    Poisson thresholds
    ~~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        PoissonThreshold([state=0])
        HomogeneousPoissonThreshold([state=0])
    
    The Poisson process gets the rates from the specified state variable, the
    homogeneous version uses the rates from the specified variable of the first
    neuron in the group.      
    """
    reinit_default_clock()

    # test that Threshold works as expected with default state
    G = NeuronGroup(3, model=LazyStateUpdater(), reset=Reset(0.), threshold=Threshold(1.), init=(0.,))
    M = SpikeMonitor(G, True)
    net = Network(G, M)
    net.run(1 * msecond)
    assert (len(M.spikes) == 0)
    G.state(0)[:] = array([0.5, 1.5, 2.5])
    net.run(1 * msecond)
    i, t = zip(*sorted(M.spikes, key=itemgetter(0)))
    assert (i == (1, 2))
    for s in t: assert (is_approx_equal(s, 1 * msecond))

    # test that Threshold works as expected with specified state
    def test_specified_state(G):
        M = SpikeMonitor(G, True)
        net = Network(G, M)
        net.run(1 * msecond)
        assert (len(M.spikes) == 0)
        net.reinit()
        G.state(0)[:] = array([0.5, 1.5, 2.5])
        net.run(1 * msecond)
        assert (len(M.spikes) == 0)
        net.reinit()
        G.state(1)[:] = array([0.5, 1.5, 2.5])
        net.run(1 * msecond)
        i, t = zip(*sorted(M.spikes, key=itemgetter(0)))
        assert (i == (1, 2))
        for s in t: assert (is_approx_equal(s, 0 * msecond))
    G = NeuronGroup(3, model=LazyStateUpdater(numstatevariables=2),
                    reset=Reset(0., state=1), threshold=Threshold(1., state=1),
                    init=(0., 0.))
    test_specified_state(G)
    # use string threshold
    eqs = '''v : 1
             w : 1
          '''
    G = NeuronGroup(3, model=eqs, reset=Reset(0., state=1), threshold='w > 1')
    test_specified_state(G)
    
    # test that VariableThreshold works as expected
    def test_variable_threshold(G):    
        M = SpikeMonitor(G, True)
        net = Network(G, M)
        get_default_clock().reinit()
        G.state(2)[:] = array([1., 2., 3.]) # the thresholds
        G.state(1)[:] = array([4., 1., 2.]) # the values
        net.run(1 * msecond)
        i, t = zip(*sorted(M.spikes, key=itemgetter(0)))
        assert (i == (0,))
        assert (is_approx_equal(t[0], 0 * second))

    G = NeuronGroup(3, model=LazyStateUpdater(numstatevariables=3),
                    reset=Reset(0., state=1),
                    threshold=VariableThreshold(2, state=1), init=(0., 0., 0.))
    test_variable_threshold(G)
    
    # use string threshold
    eqs = '''v : 1
             w : 1
             x : 1
          '''
    G = NeuronGroup(3, model=eqs, reset=Reset(0., state=1),
                    threshold='w > x')
    test_variable_threshold(G)
    
    # test that FunThreshold works as expected
    def f(S0, S1):
        return S0 > S1 * S1
    G = NeuronGroup(3, model=LazyStateUpdater(numstatevariables=2), reset=Reset(0.), threshold=FunThreshold(f), init=(0., 0.))
    G.state(0)[:] = array([2., 3., 10.])
    G.state(1)[:] = array([1., 2., 3.]) # the square root of the threshold values
    M = SpikeMonitor(G, True)
    net = Network(G, M)
    get_default_clock().reinit()
    net.run(1 * msecond)
    i, t = zip(*sorted(M.spikes, key=itemgetter(0)))
    assert (i == (0, 2))
    for s in t: assert (is_approx_equal(s, 0 * msecond))

    # test that SimpleFunThreshold works as expected
    def f(S):
        return S > 1.
    G = NeuronGroup(3, model=LazyStateUpdater(), reset=Reset(0.), threshold=SimpleFunThreshold(f), init=(0.,))
    G.state(0)[:] = array([0.5, 1.5, 2.5])
    M = SpikeMonitor(G, True)
    net = Network(G, M)
    get_default_clock().reinit()
    net.run(1 * msecond)
    i, t = zip(*sorted(M.spikes, key=itemgetter(0)))
    assert (i == (1, 2))
    for s in t: assert (is_approx_equal(s, 0 * msecond))

    # test that EmpiricalThreshold works as expected
    G = NeuronGroup(1, model=LazyStateUpdater(numstatevariables=2), reset=NoReset(), threshold=EmpiricalThreshold(1., refractory=0.5 * msecond, state=1), init=(0., 2.))
    M = SpikeMonitor(G, True)
    net = Network(G, M)
    get_default_clock().reinit()
    net.run(1.6 * msecond)
    i, t = zip(*sorted(M.spikes, key=itemgetter(1)))
    assert (i == (0, 0, 0, 0))
    for i, s in enumerate(t): assert (is_approx_equal(s, i * 0.5 * msecond))

    # test that PoissonThreshold works
    def test_poisson_threshold(G):
        init = float(1. / get_default_clock().dt) # should cause spiking at every time interval        
        G.state(0)[:] = array([0., init, 0.])
        M = SpikeMonitor(G, True)
        net = Network(G, M)
        net.run(1 * msecond)
        assert (len(M.spikes))
        i, t = zip(*sorted(M.spikes, key=itemgetter(1)))
        assert (all(j == 1 for j in i))
    
    G = NeuronGroup(3, model=LazyStateUpdater(), reset=NoReset(),
                    threshold=PoissonThreshold())
    test_poisson_threshold(G)
    
    # Poisson threshold via a string threshold using the rand() function
    eqs = '''v : 1
             w : 1
             x : 1
          '''
    
    # A threshold with rand in it is not supported by CThreshold
    if not (get_global_preference('usecodegen') and
            get_global_preference('usecodegenthreshold') and
            get_global_preference('useweave') and
            get_global_preference('usecodegenweave')):            
        G = NeuronGroup(3, model=eqs, reset=NoReset(), threshold='rand() < v')
        test_poisson_threshold(G)
        
    G = NeuronGroup(3, model=eqs, reset=NoReset(), threshold=StringThreshold('rand() < v'))
    test_poisson_threshold(G)



    # test that HomogeneousPoissonThreshold works
    init = float(1. / get_default_clock().dt) # should cause spiking at every time interval
    G = NeuronGroup(3, model=LazyStateUpdater(), reset=NoReset(), threshold=HomogeneousPoissonThreshold())
    M = SpikeMonitor(G, True)
    net = Network(G, M)
    G.state(0)[:] = array([0., init, 0.]) # should do nothing, because only first neuron is looked at 
    net.run(1 * msecond)
    assert (len(M.spikes) == 0)
    G.state(0)[:] = array([init, 0., 0.]) # should do nothing, because only first neuron is looked at 
    net.run(1 * msecond)
    # we actually cannot make any assertion about the behaviour of this system, other than
    # that it should run correctly    

if __name__ == '__main__':
    test()
