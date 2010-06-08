from brian.tools.statistics import *
from brian.units import *
from brian.stdunits import *
from numpy import *
from numpy.random import *
from nose.tools import *

def test_group_correlations():
    rate = 10.0
    n = 100
    poisson = cumsum(exponential(1 / rate, n))
    spikes = [(0, t * second) for t in poisson]
    spikes.extend([(0, t * second + 1 * ms) for t in poisson])
    spikes = sort_spikes(spikes)
    S, tauc = group_correlations(spikes, delta=3 * ms)
    assert abs(tauc[0] - .001) < .0005

if __name__ == '__main__':
    test_group_correlations()
