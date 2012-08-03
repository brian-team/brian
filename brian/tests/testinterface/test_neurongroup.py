from brian import *
from brian.utils.approximatecomparisons import *
from nose.tools import *

def test_poissongroup():
    ''' Test PoissonGroup '''
    reinit_default_clock()

    # PoissonGroup with fixed rate, test that basic state setting works
    G = PoissonGroup(10, rates=10 * Hz)
    G1 = G[:5]
    G2 = G[5:]
    G1.rate = 5 * Hz
    G2.rate = 20 * Hz
    assert(is_approx_equal(G.rate[0], float(5 * Hz)))
    assert(is_approx_equal(G1.rate[0], float(5 * Hz)))
    assert(is_approx_equal(G2.rate[0], float(20 * Hz)))
    
    # PoissonGroup with fixed rates, test 0 and 1/dt rate (the only
    # deterministic rates)
    G = PoissonGroup(2, rates=array([0.0, 1.0/defaultclock.dt]))
    counter = SpikeCounter(G)
    runtime = 10 * ms
    run(runtime)
    assert(counter.count[0] == 0)
    assert(counter.count[1] == runtime/defaultclock.dt)
    
    # Same, but with a function returning the rates
    def rate_fun(t):
        return array([0.0, 1.0/defaultclock.dt])
    
    G = PoissonGroup(2, rates=rate_fun)
    counter = SpikeCounter(G)
    runtime = 10 * ms
    run(runtime)
    assert(counter.count[0] == 0)
    assert(counter.count[1] == runtime/defaultclock.dt)


def test_linked_var():
    ''' Tests whether linking variables between groups works. '''
    
    # Test the basic linked_var behaviour
    
    reinit_default_clock()
    
    G1 = NeuronGroup(1, model='dv/dt = -v/(10*ms) : 1')
    G2 = NeuronGroup(1, model='v: 1')
    
    #link the two variables
    G2.v = linked_var(G1, 'v')
    
    G1.v = 1    
    mon1 = StateMonitor(G1, 'v', record=True)
    mon2 = StateMonitor(G2, 'v', record=True)
    
    run(50*ms)
    
    # The linked variable is updated of the beginning of a time step, before the
    # update hapenns --> The values are shifted by one
    assert(sum(abs(mon1[0][:-1] - mon2[0][1:])) == 0)

    # Test the basic linked_var behaviour with when='middle'

    reinit_default_clock()

    #link the two variables, copy after the update step
    G2.v = linked_var(G1, 'v', when='middle')
    
    G1.v = 1    
    mon1 = StateMonitor(G1, 'v', record=True)
    mon2 = StateMonitor(G2, 'v', record=True)
    
    run(50*ms)
    
    assert(sum(abs(mon1[0] - mon2[0])) == 0)    

    # Test the basic linked_var behaviour with an applied function

    reinit_default_clock()

    #link the two variables, copy after the update step
    G2.v = linked_var(G1, 'v', when='middle', func=lambda x: x * 2)
    
    G1.v = 1    
    mon1 = StateMonitor(G1, 'v', record=True)
    mon2 = StateMonitor(G2, 'v', record=True)
    
    run(50*ms)
    
    assert(sum(abs(2 * mon1[0] - mon2[0])) == 0)    
    

def test_variable_setting():
    '''
    Test that assigning to state variables and parameters works but assigning
    to static equations is forbidden.
    '''
    eqs = '''
    dv/dt = -v / tau : 1
    vsquared = v ** 2 : 1
    v_alias = v
    tau : second
    tau_alias = tau
    '''
    G = NeuronGroup(1, model=eqs)
    # Assigning to a state (differential equation)
    G.v = 0.25
    assert G.v == 0.25 and G.v_alias == 0.25
    # indirectly via an alias
    G.v_alias = 0.5
    assert G.v == 0.5 and G.v_alias == 0.5
    # Assigning to a parameter
    G.tau = 5 * ms
    assert G.tau == 5 * ms and G.tau_alias == 5 * ms
    # via an alias
    G.tau_alias = 50 * ms
    assert G.tau == 50 * ms and G.tau_alias == 50 * ms

    # assert that assigning to a static equation does not work
    def assign_to_static():
        G.vsquared = 2
    assert_raises(ValueError, assign_to_static)


if __name__ == '__main__':
    test_poissongroup()
    test_linked_var()
    test_variable_setting()