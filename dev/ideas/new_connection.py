from brian import *
import numpy
from itertools import izip

colon_slice = slice(None,None,None)

class ConnectionVector(object):
    pass
#    def __array__(self, dtype=None):
#        return NotImplemented
#    def __len__(self):
#        return NotImplemented

class DenseConnectionVector(ConnectionVector, numpy.ndarray):
    def __new__(subtype, arr):
        #return numpy.array(arr, dtype=float).view(subtype)
        return numpy.array(arr).view(subtype)
#    def __array__(self, dtype=None):
#        if dtype==None: dtype=self.dtype
#        return numpy.ndarray.__array__(self, dtype)
#    def __str__(self):
#        self = asarray(self)
#        return str(self)
#    def __repr__(self):
#        self = asarray(self)
#        return repr(self)

class SparseConnectionVector(ConnectionVector, numpy.ndarray):
    def __new__(subtype, n, ind, data):
        return numpy.array(data).view(subtype)
    def __init__(self, n, ind, data):
        self.n = n
        self.ind = ind
    def __array_wrap__(self, obj, context=None):
        print 'wrapping'
        if context is not None:
            ufunc = context[0]
            args = context[1]
        x = numpy.ndarray.__array_wrap__(self, obj, context)
#        if not hasattr(x,'_units') or hasattr(x,'_realunits_implied'):
#            x._units = self._units
#            del x._realunits_implied
        return x
#    def __array__(self, dtype=None):
#        if dtype==None: dtype=float
#        x = numpy.zeros(self.n, dtype=dtype)
#        x[self.ind] = self.data
#        return x
#    def __len__(self):
#        return self.n
#    def __str__(self):
#        return 'SparseConnectionVector length '+str(self.n)+', indices='+str(self.ind)+', data='+str(self.data)
#    __repr__ = __str__

class ConnectionMatrix(object):
    # methods to be implemented by subclass
    def get_row(self, i):
        return NotImplemented
    def get_row_sparse(self, i):
        return NotImplemented
    def get_row_dense(self, i):
        return NotImplemented
    def get_col(self, i):
        return NotImplemented
    def get_col_sparse(self, i):
        return NotImplemented
    def get_col_dense(self, i):
        return NotImplemented
    def set_row(self, i, x):
        return NotImplemented    
    def set_row_sparse(self, i, ind, data):
        return NotImplemented    
    def set_row_dense(self, i, x):
        return NotImplemented
    def set_col(self, i, x):
        return NotImplemented    
    def set_col_sparse(self, i, ind, data):
        return NotImplemented    
    def set_col_dense(self, i, x):
        return NotImplemented
    def set_single(self, i, j, x):
        return NotImplemented
    def get_single(self, i, j, x):
        return NotImplemented
    def freeze(self):
        pass
    def getnnz(self):
        return NotImplemented
    # TODO: something like this? need to think about this
    def get_rows_dense(self, rows):
        return NotImplemented
    def get_rows_sparse_list(self, rows):
        return NotImplemented
    # TODO: something like this?
    def insert_in_row_sparse(self, i, ind, data):
        return NotImplemented
    # methods that can be overwritten to improve performance
    def __array__(self, dtype=None):
        if dtype is None: dtype=float
        x = numpy.zeros(self.shape, dtype=dtype)
        for i in xrange(self.shape[0]):
            x[i] = self.get_row_dense(i)
        return x
    # utility methods
    def get_row_sparse_from_dense(self, i):
        x = self.get_row_dense(i)
        ind = where(x>0)[0]
        return SparseConnectionVector(len(x), ind, x[ind])
    def get_row_dense_from_sparse(self, i):
        return DenseConnectionVector(array(self.get_row_sparse(i)))
    def get_col_sparse_from_dense(self, i):
        x = self.get_col_dense(i)
        ind = where(x>0)[0]
        return SparseConnectionVector(len(x), ind, x[ind])
    def get_col_dense_from_sparse(self, i):
        return DenseConnectionVector(array(self.get_col_sparse(i)))
    def set_row_sparse_with_dense(self, i, x):
        ind = where(x>0)[0]
        self.set_row_sparse(i, ind, x[ind])
    def set_row_dense_with_sparse(self, i, ind, data):
        self.set_row_dense(i, array(SparseConnectionVector(self.shape[1], ind, data)))
    def set_col_sparse_with_dense(self, i, x):
        ind = where(x>0)[0]
        self.set_col_sparse(i, ind, x[ind])
    def set_col_dense_with_sparse(self, i, ind, data):
        self.set_col_dense(i, array(SparseConnectionVector(self.shape[0], ind, data)))
    # TODO: emulation of numpy indexing needs to be improved and semantics decided
    # emulation of numpy indexing
    # we support the following indexing schemes:
    # - s[:]
    # - s[i,:]
    # - s[:,i]
    def __getitem__(self, item):
        if isinstance(item,tuple) and isinstance(item[0],int) and item[1]==colon_slice:
            return self.get_row(item[0])
        if isinstance(item,slice):
            if item==colon_slice:
                return self
            else:
                raise ValueError(str(item)+' not supported.')
        if isinstance(item,int):
            return self.get_row(item)
        if isinstance(item,tuple):
            if len(item)!=2:
                raise TypeError('Only 2D indexing supported.')
            item_i, item_j = item
            if isinstance(item_i, int) and isinstance(item_j, slice):
                if item_j==colon_slice:
                    return self.get_row(item_i)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, slice) and isinstance(item_j, int):
                if item_i==colon_slice:
                    return self.get_col(item_j)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, int) and isinstance(item_j, int):
                raise ValueError('Do not support (i,j) indexing.')
#                pointer = self.get_pointer(item_i, item_j)
#                if pointer is None:
#                    return 0.0
#                return self.alldata[pointer]
            raise TypeError('Only (i,:), (:,j) indexing supported.')
        raise TypeError('Can only get items of type slice or tuple')

class DenseConnectionMatrix(ConnectionMatrix, numpy.ndarray):
    def __init__(self, shape, **kwds):
        numpy.ndarray.__init__(self, shape, **kwds)
        self[:]=0
    def get_row_dense(self, i):
        return DenseConnectionVector(self[i])
    get_row = get_row_dense
    get_row_sparse = ConnectionMatrix.get_row_sparse_from_dense
    def get_col_dense(self, i):
        return DenseConnectionVector(self[:,i])
    get_col = get_col_dense
    get_col_sparse = ConnectionMatrix.get_col_sparse_from_dense
    def set_row_dense(self, i, x):
        self[i] = array(x)
    set_row = set_row_dense
    set_row_sparse = ConnectionMatrix.set_row_dense_with_sparse
    def set_col_dense(self, i, x):
        self[:, i] = array(x)
    set_col = set_col_dense
    set_col_sparse = ConnectionMatrix.set_col_dense_with_sparse
    def getnnz(self):
        return sum(self[:]!=0)

class SparseConnectionMatrix(ConnectionMatrix):
    def __init__(self, shape, **kwds):
        self.shape = shape
        self.rowj = [array([], dtype=int)]*shape[0]
        self.rowdata = [array([], dtype=float)]*shape[0]
    def __array__(self, dtype=None):
        if dtype is None: dtype=self.rowdata[0].dtype
        x = numpy.zeros(self.shape, dtype=dtype)
        for i in xrange(self.shape[0]):
            x[i] = self.get_row_dense(i)
        return x
    def getnnz(self):
        return sum(len(r) for r in self.rowj)
    def get_row_sparse(self, i):
        return SparseConnectionVector(self.shape[1], self.rowj[i], self.rowdata[i])
    get_row = get_row_sparse
    get_row_dense = ConnectionMatrix.get_row_dense_from_sparse
    def set_row_sparse(self, i, ind, data):
        if len(self.rowj[i]):
            raise RuntimeError('Cannot set row value twice for current implementation.')
        self.rowj[i] = array(ind, dtype=int)
        self.rowdata[i] = array(data, dtype=float)
    def set_row(self, i, x):
        if isinstance(x, SparseConnectionVector):
            self.set_row_sparse(i, x.ind, x.data)
        else:
            self.set_row_dense(i, x)
    set_row_dense = ConnectionMatrix.set_row_sparse_with_dense

# TODO: finish FixedNNZSparseConnectionMatrix
# Issues: insertion/deletion, how to handle it and stay efficient
# Note: current data storage requirement is 14 bytes per entry + fixed overhead
class FixedNNZSparseConnectionMatrix(ConnectionMatrix):
    def __init__(self, shape, nnzmax, **kwds):
        self.shape = shape
        self.nnzmax = nnzmax
        self.nnz = 0
        self.alldata = numpy.zeros(nnzmax)
        self.unusedinds = numpy.arange(nnzmax, dtype=int) # pop from the end
#        self.alli = numpy.zeros(nnzmax, dtype=int)
#        self.allj = numpy.zeros(nnzmax, dtype=int)
        self.rowj = [array([], dtype=int)]*shape[0]
        self.rowdataind = [array([], dtype=int)]*shape[0]
        self.coli = [array([], dtype=int)]*shape[0]
        self.coldataind = [array([], dtype=int)]*shape[0]
    def getnnz(self):
        return self.nnz
    def _alloc_indices(self, n):
        if self.nnz + n > self.nnzmax:
            raise MemoryError('Not enough space to store more entries in connection matrix, increase nnzmax.')
        return self.unusedinds[(self.nnzmax-self.nnz-n):(self.nnzmax-self.nnz)]
    def set_row_sparse(self, i, ind, data):
        if len(self.rowj[i]):
            raise RuntimeError('Cannot set row value twice for current implementation.')
        newinds = self._alloc_indices(len(ind))
        self.alldata[newinds] = data
#        self.alli[newinds] = i
#        self.allj[newinds] = ind
        self.rowdataind[i] = copy(newinds)
        self.rowj[i] = array(ind, dtype=int)
        self.nnz += len(ind)
        for c, adp in izip(ind, newinds):
            self.coli[c] = numpy.append(self.coli[c], i)
            self.coldataind[c] = numpy.append(self.coldataind[c], adp)
    def get_row_sparse(self, i):
        return SparseConnectionVector(self.shape[1], self.rowj[i], self.alldata[self.rowdataind[i]])
    get_row = get_row_sparse
    get_row_dense = ConnectionMatrix.get_row_dense_from_sparse
    def get_col_sparse(self, j):
        return SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataind[j]])
    get_col = get_col_sparse
    get_col_dense = ConnectionMatrix.get_col_dense_from_sparse

class FrozenSparseConnectionMatrix(ConnectionMatrix):
    def __init__(self, val):
        self.nnz = nnz = val.getnnz()
        alldata = numpy.zeros(nnz)
        alli = numpy.zeros(nnz, dtype=int)
        allj = numpy.zeros(nnz, dtype=int)
        rowind = numpy.zeros(val.shape[0]+1, dtype=int)
        rowdata = []
        rowj = []
        coli = []
        coldataindices = []
        i = 0
        for c in xrange(val.shape[0]):
            sr = val.get_row_sparse(c)
            r = sr.ind
            d = sr.data
            rowind[c] = i
            alldata[i:i+len(d)] = d
            allj[i:i+len(r)] = r
            alli[i:i+len(r)] = c
            rowdata.append(alldata[i:i+len(d)])
            rowj.append(allj[i:i+len(r)])
            i = i+len(r)
        rowind[val.shape[0]] = i
        counts = numpy.histogram(allj, numpy.arange(val.shape[1]+1, dtype=int), new=True)[0]
        curcdi = numpy.zeros(val.shape[1], dtype=int)
        for j in xrange(val.shape[1]):
            coldataindices.append(numpy.zeros(counts[j], dtype=int))
        for i, j in enumerate(allj):
            coldataindices[j][curcdi[j]] = i
            curcdi[j]+=1
        for j in xrange(val.shape[1]):
            coli.append(alli[coldataindices[j]])
        self.alldata = alldata
        self.rowdata = rowdata
        self.allj = allj
        self.alli = alli
        self.rowj = rowj
        self.coli = coli
        self.coldataindices = coldataindices
        self.rowind = rowind
        self.shape = val.shape
#        self.rowret = numpy.zeros(val.shape[1])
#        self.colret = numpy.zeros(val.shape[0])
#        self.rowset = numpy.zeros(val.shape[1])
#        self.colset = numpy.zeros(val.shape[0])
    def __array__(self, dtype=None):
        if dtype==None: dtype=self.alldata.dtype
        x = numpy.zeros(self.shape)
        for i in xrange(self.shape[0]):
            x[i] = array(self.get_row_dense(i), dtype=dtype)
        return x
    def getnnz(self):
        return self.nnz
    def get_row_sparse(self, i):
        return SparseConnectionVector(self.shape[1], self.rowj[i], self.rowdata[i])
    get_row = get_row_sparse
    get_row_dense = ConnectionMatrix.get_row_dense_from_sparse
    def get_col_sparse(self, j):
        return SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataindices[j]])
    get_col = get_col_sparse
    get_col_dense = ConnectionMatrix.get_col_dense_from_sparse
    # TODO: decide set_row/set_col semantics
#    def set_row(self, i, val):
#        self.rowset[:] = val
#        self.rowdata[i][:] = self.rowset[self.rowj[i]]
#    def set_col(self, j, val):
#        self.colset[:] = val
#        self.alldata[self.coldataindices[j]] = self.colset[self.coli[j]]

if __name__=='__main__':
    x = SparseConnectionVector(10, [1,2], [3.,4.])
    y = randn(10)
    #print x*x
    print x+y