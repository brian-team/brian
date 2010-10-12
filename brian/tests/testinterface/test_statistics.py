from brian import *
from nose.tools import *

def test():
    """
    Statistics module
    """
    assert total_correlation([],[]) is NaN
    assert correlogram([],[]) is NaN
    assert firing_rate([]) is NaN
    assert CV([]) is NaN
    
if __name__ == '__main__':
    test()
