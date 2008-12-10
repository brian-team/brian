from brian import *
import numpy
from itertools import izip

colon_slice = slice(None,None,None)

def todense(x):
    if hasattr(x, 'todense'):
        return x.todense()
    return array(x)

class ConnectionVector(object):
    def todense(self):
        return NotImplemented
    def tosparse(self):
        return NotImplemented

class DenseConnectionVector(ConnectionVector, numpy.ndarray):
    def __new__(subtype, arr):
        return numpy.array(arr).view(subtype)
    def todense(self):
        return self
    def tosparse(self):
        return SparseConnectionVector(len(self), self.nonzero(), self)

class SparseConnectionVector(ConnectionVector, numpy.ndarray):
    def __new__(subtype, n, ind, data):
        x = numpy.array(data).view(subtype)
        x.n = n
        x.ind = ind
        return x
    def __array_finalize__(self, orig):
        try:
            self.ind = orig.ind
            self.n = orig.n
        except AttributeError:
            pass
        return self
    def todense(self):
        x = zeros(self.n)
        x[self.ind] = self
        return x
    def tosparse(self):
        return self
    modifymeths = ['__add__', '__and__', 
         '__div__', '__divmod__', '__eq__',
         '__floordiv__', '__ge__', '__gt__', '__iadd__', '__iand__', '__idiv__',
         '__ifloordiv__', '__ilshift__', '__imod__', '__imul__',
         '__ior__', '__ipow__', '__irshift__', '__isub__', '__itruediv__',
         '__ixor__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__',
         '__ne__', '__or__', '__pow__', '__radd__', '__rand__', '__rdiv__',
         '__rdivmod__', '__rfloordiv__', '__rlshift__',
         '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__',
         '__rsub__', '__rtruediv__', '__rxor__', '__sub__', '__truediv__', '__xor__']
    template = '''
def __add__(self, other):
    if isinstance(other, SparseConnectionVector):
        if other.ind is not self.ind:
            raise TypeError('__add__(SparseConnectionVector, SparseConnectionVector) only defined if indices are the same')
        return SparseConnectionVector(self.n, self.ind, numpy.ndarray.__add__(asarray(self), asarray(other)))
    return SparseConnectionVector(self.n, self.ind, numpy.ndarray.__add__(asarray(self), other[self.ind]))
'''.strip()
    for m in modifymeths:
        s = template.replace('__add__', m)
        exec s
    del modifymeths, template

class ConnectionMatrix(object):
    # methods to be implemented by subclass
    def get_row(self, i):
        return NotImplemented
    def get_col(self, i):
        return NotImplemented
    def set_row(self, i, x):
        return NotImplemented    
    def set_col(self, i, x):
        return NotImplemented    
    def set_element(self, i, j, x):
        return NotImplemented
    def get_element(self, i, j):
        return NotImplemented
    def get_rows(self, rows):
        return NotImplemented
    def insert(self, i, j, x):
        return NotImplemented
    def getnnz(self):
        return NotImplemented
    def todense(self):
        return array([todense(r) for r in self])
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
                return self.get_element(item_i, item_j)
            raise TypeError('Only (i,:), (:,j), (i,j) indexing supported.')
        raise TypeError('Can only get items of type slice or tuple')
    # TODO: __setitem__ as above

class DenseConnectionMatrix(ConnectionMatrix, numpy.ndarray):
    def __init__(self, shape, **kwds):
        numpy.ndarray.__init__(self, shape, **kwds)
        self[:]=0
    def get_row(self, i):
        return DenseConnectionVector(self[i])
    def get_col(self, i):
        return DenseConnectionVector(self[:,i])
    def set_row(self, i, x):
        self[i] = todense(x)
    def set_col(self, i, x):
        self[:, i] = todense(x)

# TODO: test SparseConnectionMatrix
class SparseConnectionMatrix(ConnectionMatrix):
    def __init__(self, val, column_access=True):
        self.nnz = nnz = val.getnnz()
        alldata = numpy.zeros(nnz)
        if column_access:
            alli = numpy.zeros(nnz, dtype=int)
        allj = numpy.zeros(nnz, dtype=int)
        rowind = numpy.zeros(val.shape[0]+1, dtype=int)
        rowdata = []
        rowj = []
        if column_access:
            coli = []
            coldataindices = []
        i = 0
        for c in xrange(val.shape[0]):
            sr = val[c, :]
            sr = sr.tolil()
            r = sr.rows[0]
            d = sr.data[0]
            rowind[c] = i
            alldata[i:i+len(d)] = d
            allj[i:i+len(r)] = r
            if column_access:
                alli[i:i+len(r)] = c
            rowdata.append(alldata[i:i+len(d)])
            rowj.append(allj[i:i+len(r)])
            i = i+len(r)
        rowind[val.shape[0]] = i
        if column_access:
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
        self.rowj = rowj
        self.rowind = rowind
        self.shape = val.shape
        self.column_access = column_access
        if column_access:
            self.alli = alli
            self.coli = coli
            self.coldataindices = coldataindices
    def getnnz(self):
        return self.nnz
    def get_row(self, i):
        return SparseConnectionVector(self.shape[1], self.rowj[i], self.rowdata[i])
    def get_col(self, j):
        if self.column_access:
            return SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataindices[j]])
        else:
            raise TypeError('No column access.')
    def set_row(self, i, val):
        if isinstance(val, SparseConnectionVector):
            if val.ind is not self.rowj[i]:
                if not (val.ind==self.rowj[i]).all():
                    raise ValueError('Sparse row setting must use same indices.')
            self.rowdata[i][:] = val
        else:
            val = asarray(val)
            self.rowdata[i][:] = val[self.rowj[i]]
    def set_col(self, j, val):
        if self.column_access:
            if isinstance(val, SparseConnectionVector):
                if val.ind is not self.coli[j]:
                    if not (val.ind==self.coli[j]).all():
                        raise ValueError('Sparse col setting must use same indices.')
                self.alldata[self.coldataindices[j]] = val
            else:
                val = asarray(val)
                self.alldata[self.coldataindices[j]] = val[self.coli[j]]
        else:
            raise TypeError('No column access.')

# TODO: finish DynamicConnectionMatrix
# Issues: insertion/deletion, how to handle it and stay efficient
class DynamicConnectionMatrix(ConnectionMatrix):
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

if __name__=='__main__':
    x = SparseConnectionVector(10, [1,2], [3.,4.])
    z = SparseConnectionVector(10, [5,6], [7.,8.])
    y = randn(10)
    #print x*x
    print (x+y)+x
    print 2*asarray(x)+y[x.ind]
    print x*y
    print asarray(x)*y[x.ind]
    #print x*z
    import time
    n = 2000
    m = 1000
    x = SparseConnectionVector(n, arange(m), arange(m, dtype=float))
    y = randn(n)
    z = asarray(x)
    start = time.time()
    for _ in xrange(10000):
        x+y
    print time.time()-start
    start = time.time()
    for _ in xrange(10000):
        z+y[x.ind]
    print time.time()-start
#    x = array([1])
#    meths = [m for m in dir(x) if m.startswith('__') and callable(getattr(x,m))]
#    twoargs = []
#    for m in meths:
#        try:
#            getattr(x,m)(x)
#            twoargs.append(m)
#        except Exception, e:
#            print m, e
#    print
#    print twoargs