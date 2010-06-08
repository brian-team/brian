from scipy import sparse
import numpy
from itertools import izip
from operator import isSequenceType
import bisect
from numpy import ndarray
import itertools


class lil_matrix(sparse.lil_matrix):
    # Unfortunately we still need to implement this because although scipy 0.7.0
    # now supports X[a:b,c:d] for sparse X it is unbelievably slow (shabby code
    # on their part).
    def __setitem__(self, index, W):
        """
        Speed-up if x is a sparse matrix.
        TODO: checks (first remove the data).
        TODO: once we've got this working in all cases, should we submit to scipy?
        """
        try:
            i, j = index
        except (ValueError, TypeError):
            raise IndexError, "invalid index"

        if isinstance(i, slice) and isinstance(j, slice) and\
               (i.step is None) and (j.step is None) and\
               (isinstance(W, sparse.lil_matrix) or isinstance(W, numpy.ndarray)):
            rows = self.rows[i]
            datas = self.data[i]
            j0 = j.start
            if isinstance(W, sparse.lil_matrix):
                for row, data, rowW, dataW in izip(rows, datas, W.rows, W.data):
                    jj = bisect.bisect(row, j0) # Find the insertion point
                    row[jj:jj] = [j0 + k for k in rowW]
                    data[jj:jj] = dataW
            elif isinstance(W, ndarray):
                nq = W.shape[1]
                for row, data, rowW in izip(rows, datas, W):
                    jj = bisect.bisect(row, j0) # Find the insertion point
                    row[jj:jj] = range(j0, j0 + nq)
                    data[jj:jj] = rowW
        elif isinstance(i, slice) and isinstance(j, int) and isSequenceType(W):
            # This corrects a bug in scipy sparse matrix as of version 0.7.0, but
            # it is not efficient!
            for w, k in izip(W, xrange(*i.indices(self.shape[0]))):
                sparse.lil_matrix.__setitem__(self, (k, j), w)
        elif isinstance(i, int) and isSequenceType(j) and len(j) == 0:
            return
        else:
            sparse.lil_matrix.__setitem__(self, index, W)
