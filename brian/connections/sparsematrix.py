from base import *

__all__ = ['SparseMatrix', 'sparse']

use_sparse_matrix = 'scipy_patch' # values are own, own_scipy, scipy, scipy_patch
if use_sparse_matrix == 'own':
    from ..utils import sparse_matrix as sparse
elif use_sparse_matrix == 'own_scipy':
    from ..utils import sparse # Brian's version of scipy sparse matrix library
elif use_sparse_matrix == 'scipy_patch':
    from ..utils import sparse_patch as sparse

# set this to True using the sparse library packaged with Brian, from scipy 0.7.1
if use_sparse_matrix == 'own' or use_sparse_matrix == 'own_scipy':
    oldscipy = True
else:
    oldscipy = scipy.__version__.startswith('0.6.') or scipy.__version__.startswith('0.7.1')

if use_sparse_matrix == 'own' or use_sparse_matrix == 'scipy_patch':
    SparseMatrix = sparse.lil_matrix
else:
    class SparseMatrix(sparse.lil_matrix):
        '''
        Used as the base for sparse construction matrix classes, essentially just scipy's lil_matrix.
        
        The scipy lil_matrix class allows you to specify slices in ``__setitem__`` but the
        performance is cripplingly slow. This class has a faster implementation.
        '''
        # Unfortunately we still need to implement this because although scipy 0.7.0
        # now supports X[a:b,c:d] for sparse X it is unbelievably slow (shabby code
        # on their part).
        def __setitem__(self, index, W):
            """
            Speed-up if x is a sparse matrix.
            TODO: checks (first remove the data).
            """
            try:
                i, j = index
            except (ValueError, TypeError):
                raise IndexError, "invalid index"

            if isinstance(i, slice) and isinstance(j, slice) and\
               (i.step is None) and (j.step is None) and\
               (isinstance(W, sparse.spmatrix) or isinstance(W, numpy.ndarray)):
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
            elif oldscipy and isinstance(i, int) and isinstance(j, (list, tuple, numpy.ndarray)):
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
            else:
                sparse.lil_matrix.__setitem__(self, index, W)
