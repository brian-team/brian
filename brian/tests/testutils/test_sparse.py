from brian.connection import SparseMatrix
from nose.tools import *

def test_sparse_matrix():
    x = SparseMatrix((10, 10))
    x[0, 4] = 3
    x[1, :] = 5
    x[2, []] = []
    x[3, [0]] = [6]
    x[4, [0, 1]] = [7, 8]
    x[5, 2:4] = [1,2]
    x[6:8, 4:7] = 1

if __name__=='__main__':
    test_sparse_matrix()
    