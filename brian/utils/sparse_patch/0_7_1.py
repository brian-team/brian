from scipy import sparse
import numpy
import itertools
from itertools import izip, repeat
from operator import isSequenceType, isNumberType
import bisect
from numpy import ndarray


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
        elif isinstance(i, int) and isinstance(j, (list, tuple, numpy.ndarray)):
            if len(j) == 0:
                return
            row = dict(izip(self.rows[i], self.data[i]))
            try:
                row.update(dict(izip(j, W)))
            except TypeError:
                row.update(dict(izip(j, itertools.repeat(W))))
            items = row.items()
            items.sort()
            row, data = izip(*items)
            self.rows[i] = list(row)
            self.data[i] = list(data)
        elif isinstance(i, slice) and isinstance(j, int) and isSequenceType(W):
            # This corrects a bug in scipy sparse matrix as of version 0.7.0, but
            # it is not efficient!
            for w, k in izip(W, xrange(*i.indices(self.shape[0]))):
                sparse.lil_matrix.__setitem__(self, (k, j), w)
        elif isinstance(i, int) and isinstance(j, slice) and (isNumberType(W) and not isSequenceType(W)):
            # this fixes a bug in scipy 0.7.1
            sparse.lil_matrix.__setitem__(self, index, [W] * len(xrange(*j.indices(self.shape[1]))))
        elif isinstance(i, slice) and isinstance(j, slice) and isNumberType(W):
            n = len(xrange(*i.indices(self.shape[0])))
            m = len(xrange(*j.indices(self.shape[1])))
            sparse.lil_matrix.__setitem__(self, index, W * numpy.ones((n, m)))
        else:
            sparse.lil_matrix.__setitem__(self, index, W)
