'''
Attempt to write our own sparse matrix class - it doesn't need much sophistication.

TODO:
* __getitem__ not implemented
* Cannot set rows/cols using another sparse matrix (essential for Brian)
* Highly inefficient, as it just reduces everything to a sequence of x[i,j]=v
  operations.
'''

from numpy import *
from operator import isNumberType, isSequenceType
from itertools import repeat
import bisect

__all__ = ['lil_matrix']

class lil_matrix(object):
    def __init__(self, arg):
        if isinstance(arg, lil_matrix):
            raise ValueError('Cannot yet initialise from lil_matrix')
        elif isinstance(arg, tuple):
            M, N = arg
            self.rows = [[] for _ in xrange(M)]
            self.data = [[] for _ in xrange(M)]
            self.shape = (M, N)
            self.dtype = float
        else:
            raise ValueError('Cannot yet initialise from dense matrix')
    def __len__(self):
        M, N = self.shape
        if M>1: return M
        else: return N
    def __getitem__(self, item):
        if isinstance(item, tuple):
            pass
        elif isinstance(item, int):
            pass
        else:
            raise IndexError("Don't know what to do with that index.")
    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            if len(item)==2:
                i, j = item
                M, N = self.shape
                if isinstance(i, int) and isinstance(j, int):
                    if not isNumberType(value):
                        raise ValueError('Setting single item with a non-scalar.')
                    if not (0<=i<M and 0<=j<N):
                        raise IndexError('Index (%d, %d) out of range (%d, %d)' % (i, j, M, N))
                    if not value==0:
                        row = self.rows[i]
                        data = self.data[i]
                        k = bisect.bisect_left(row, j)
                        if k>=len(row):
                            row.append(j)
                            data.append(value)
                        elif row[k]==j:
                            data[k] = value
                        else:
                            row.insert(k, j)
                            data.insert(k, value)
                elif isinstance(i, int) and isinstance(j, (list, tuple, ndarray, slice)):
                    if isinstance(j, slice):
                        j = range(*j.indices(N))
                    if not isSequenceType(value):
                        value = repeat(value)
                    elif not len(value)==len(j):
                        raise ValueError('Shape mismatch.')
                    for k, v in zip(j, value):
                        self[i, k] = v
                elif isinstance(i, (list, tuple, ndarray, slice)) and isinstance(j, int):
                    if isinstance(i, slice):
                        i = range(*i.indices(N))
                    if not isSequenceType(value):
                        value = repeat(value)
                    elif not len(value)==len(i):
                        raise ValueError('Shape mismatch.')
                    for k, v in zip(i, value):
                        self[k, j] = v
                elif isinstance(i, (list, tuple, ndarray, slice)) and isinstance(j, (list, tuple, ndarray, slice)):
                    if isinstance(i, slice) or isinstance(j, slice):
                        if isinstance(i, slice):
                            i = range(*i.indices(N))
                        if isinstance(j, slice):
                            j = range(*j.indices(N))
                        if not isSequenceType(value):
                            value = repeat(value)
                        elif not len(value)==len(i):
                            raise ValueError('Shape mismatch.')
                        for k, v in zip(i, value):
                            self[k, j] = v
                    else:
                        if not len(i)==len(j):
                            raise ValueError('Shape mismatch.')
                        if not isSequenceType(value):
                            value = repeat(value)
                        elif not len(value)==len(j):
                            raise ValueError('Shape mismatch.')
                        for ii, jj, v in zip(i, j, value):
                            self[ii, jj] = v
                else:
                    raise IndexError("Don't know what to do with that index.")
            else:
                raise IndexError('Shape error, two dimensional only.')
        elif isinstance(item, (int, list, tuple, ndarray, slice)):
            self[item, :] = value
        else:
            raise IndexError("Don't know what to do with that index.")
    def todense(self):
        X = zeros(self.shape, dtype=self.dtype)
        for i, (row, data) in enumerate(zip(self.rows, self.data)):
            X[i, row] = data
        return X
            

def test_lil():
    x = lil_matrix((10, 10))
    test_calls = '''
    x[0, 4] = 3
    x[1, :] = 5
    x[2, []] = []
    x[3, [0]] = [6]
    x[4, [0, 1]] = [7, 8]
    x[5, 2:4] = [1,2]
    x[6:8, 4:6] = 1
    '''
    for line in test_calls.split('\n'):
        line = line.strip()
        if line:
            try:
                exec line
                print x.todense()
                print '*****'
            except Exception, e:
                print 'Line:', line
                print repr(e)
                print
    
if __name__=='__main__':
    test_lil()
