from brian import *
from brian.utils.approximatecomparisons import *
from nose.tools import *

def test_poissongroup():
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
    

if __name__ == '__main__':
    test_poissongroup()
