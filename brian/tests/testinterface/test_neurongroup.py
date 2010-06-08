from brian import *
from brian.utils.approximatecomparisons import *
from nose.tools import *

def test():
    reinit_default_clock()

    # PoissonGroup
    G = PoissonGroup(10, rates=10 * Hz)
    G1 = G[:5]
    G2 = G[5:]
    G1.rate = 5 * Hz
    G2.rate = 20 * Hz
    assert(is_approx_equal(G.rate[0], float(5 * Hz)))
    assert(is_approx_equal(G1.rate[0], float(5 * Hz)))
    assert(is_approx_equal(G2.rate[0], float(20 * Hz)))

if __name__ == '__main__':
    test()
