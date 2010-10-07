from brian.connections import SparseMatrix
from numpy import zeros, array
from nose.tools import *

def test_sparse_matrix():
    x = SparseMatrix((10, 10))
    z = zeros((10, 10), dtype=int)
    x[0, 4] = 3
    x[1, :] = 5
    x[2, []] = []
    x[3, [0]] = [6]
    x[4, [0, 1]] = [7, 8]
    x[5, 2:4] = [1, 2]
    x[6:8, 4:7] = 1
    # same thing with z to compare
    z[0, 4] = 3
    z[1, :] = 5
    z[2, []] = []
    z[3, [0]] = [6]
    z[4, [0, 1]] = [7, 8]
    z[5, 2:4] = [1, 2]
    z[6:8, 4:7] = 1
    # check the values are correct
    y = array(x.todense(), dtype=int)
    assert (y==z).all()

if __name__ == '__main__':
    test_sparse_matrix()
