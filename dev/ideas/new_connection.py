from brian import *
import numpy
import scipy
from itertools import izip
import bisect
from scipy import weave

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
        return numpy.array(arr, copy=False).view(subtype)
    def todense(self):
        return self
    def tosparse(self):
        return SparseConnectionVector(len(self), self.nonzero(), self)

class SparseConnectionVector(ConnectionVector, numpy.ndarray):
    '''
    Sparse vector class
    
    A sparse vector is typically a row or column of a sparse matrix. This
    class can be treated in many cases as if it were just a vector without
    worrying about the fact that it is sparse. For example, if you write
    ``2*v`` it will evaluate to a new sparse vector. There is one aspect
    of the semantics which is potentially confusing. In a binary operation
    with a dense vector such as ``sv+dv`` where ``sv`` is sparse and ``dv``
    is dense, the result will be a sparse vector with zeros where ``sv``
    has zeros, the potentially nonzero elements of ``dv`` where ``sv`` has
    no entry will be simply ignored. It is for this reason that it is a
    ``SparseConnectionVector`` and not a general ``SparseVector``, because
    these semantics make sense for rows and columns of connection matrices
    but not in general.
    
    Implementation details:
    
    The underlying numpy array contains the values, the attribute ``n`` is
    the length of the sparse vector, and ``ind`` is an array of the indices
    of the nonzero elements.
    '''
    def __new__(subtype, n, ind, data):
        x = numpy.array(data, copy=False).view(subtype)
        x.n = n
        x.ind = ind
        return x
    def __array_finalize__(self, orig):
        # the array is passed through this function after standard numpy operations,
        # this ensures that the indices are kept from the original array. This makes,
        # for example, sin(x) do the right thing for x a sparse vector.
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
    # This is a list of the binary operations that numpy arrays support.
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
    # This template function (where __add__ is replaced by any of the methods above) implements
    # the semantics described in this class' docstring when operating with a dense vector.
    template = '''
def __add__(self, other):
    if isinstance(other, SparseConnectionVector):
        if other.ind is not self.ind:
            raise TypeError('__add__(SparseConnectionVector, SparseConnectionVector) only defined if indices are the same')
        return SparseConnectionVector(self.n, self.ind, numpy.ndarray.__add__(asarray(self), asarray(other)))
    if isinstance(other, numpy.ndarray):
        return SparseConnectionVector(self.n, self.ind, numpy.ndarray.__add__(asarray(self), other[self.ind]))
    return SparseConnectionVector(self.n, self.ind, numpy.ndarray.__add__(asarray(self), other))
'''.strip()
    # this substitutes any of the method names in the modifymeths list for __add__ in the template
    # above and then executes them, i.e. adding them as methods to the class. When the behaviour is
    # stable, this can be replaced by the explicit definitions but it may as well be left as it is for
    # the moment (it's slower at import time, but not at run time, and errors are more difficult to
    # catch when it is done like this).
    for m in modifymeths:
        s = template.replace('__add__', m)
        exec s
    del modifymeths, template

class ConstructionMatrix(object):
    '''
    Base class for construction matrices
    
    A construction matrix is used to initialise and build connection matrices.
    A ``ConstructionMatrix`` class has to implement a method
    ``connection_matrix(*args, **kwds)`` which returns a :class:`ConnectionMatrix`
    object of the appropriate type.
    '''
    def connection_matrix(self, *args, **kwds):
        return NotImplemented

class DenseConstructionMatrix(ConstructionMatrix, numpy.ndarray):
    def connection_matrix(self, *args, **kwds):
        if 'copy' not in kwds:
            kwds['copy'] = False
        return DenseConnectionMatrix(self, *args, **kwds)
    def __setitem__(self, index, W):
        # Make it work for sparse matrices
        if isinstance(W,scipy.sparse.spmatrix):
            ndarray.__setitem__(self,index,W.todense())
        else:
            ndarray.__setitem__(self,index,W)

class SparseMatrix(scipy.sparse.lil_matrix):
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

        if isinstance(i, slice) and isinstance(j,slice) and\
           (i.step is None) and (j.step is None) and\
           (isinstance(W,scipy.sparse.lil_matrix) or isinstance(W,numpy.ndarray)):
            rows = self.rows[i]
            datas = self.data[i]
            j0=j.start
            if isinstance(W,scipy.sparse.lil_matrix):
                for row,data,rowW,dataW in izip(rows,datas,W.rows,W.data):
                    jj=bisect.bisect(row,j0) # Find the insertion point
                    row[jj:jj]=[j0+k for k in rowW]
                    data[jj:jj]=dataW
            elif isinstance(W,ndarray):
                nq=W.shape[1]
                for row,data,rowW in izip(rows,datas,W):
                    jj=bisect.bisect(row,j0) # Find the insertion point
                    row[jj:jj]=range(j0,j0+nq)
                    data[jj:jj]=rowW
        else:
            scipy.sparse.lil_matrix.__setitem__(self,index,W)

class SparseConstructionMatrix(ConstructionMatrix, SparseMatrix):
    def connection_matrix(self, *args, **kwds):
        return SparseConnectionMatrix(self, *args, **kwds)
            
class DynamicConstructionMatrix(ConstructionMatrix, SparseMatrix):
    def connection_matrix(self, *args, **kwds):
        return DynamicConnectionMatrix(self, *args, **kwds)

# this is used to look up str->class conversions for structure=... keyword
construction_matrix_register = {
        'dense':DenseConstructionMatrix,
        'sparse':SparseConstructionMatrix,
        'dynamic':DynamicConstructionMatrix,
        }

class ConnectionMatrix(object):
    '''
    Base class for connection matrix objects
    
    Connection matrix objects support a subset of the following methods:
    
    ``get_row(i)``, ``get_col(i)``
        Returns row/col ``i`` as a :class:`DenseConnectionVector` or
        :class:`SparseConnectionVector` as appropriate for the class.
    ``set_row(i, val)``, ``set_col(i, val)``
        Sets row/col with an array, :class:`DenseConnectionVector` or
        :class:`SparseConnectionVector` (if supported).
    ``get_element(i, j)``, ``set_element(i, j, val)``
        Gets or sets a single value.
    ``get_rows(rows)``
        Returns a list of rows, should be implemented without Python
        function calls for efficiency if possible.
    ``insert(i,j,x)``, ``remove(i,j)``
        For sparse connection matrices which support it, insert a new
        entry or remove an existing one.
    ``getnnz()``
        Return the number of nonzero entries.
    ``todense()``
        Return the matrix as a dense array.
    
    The ``__getitem__`` and ``__setitem__`` methods are implemented by
    default, and automatically select the appropriate methods from the
    above in the cases where the item to be got or set is of the form
    ``:``, ``i,:``, ``:,j`` or ``i,j``.
    '''
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
        return [self.get_row(i) for i in rows]
    
    def insert(self, i, j, x):
        return NotImplemented
    
    def remove(self, i, j):
        return NotImplemented
    
    def getnnz(self):
        return NotImplemented
    
    def todense(self):
        return array([todense(r) for r in self])
    # we support the following indexing schemes:
    # - s[:]
    # - s[i,:]
    # - s[:,i]
    # - s[i,j]
    
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
    
    def __setitem__(self, item, value):
        if isinstance(item,tuple) and isinstance(item[0],int) and item[1]==colon_slice:
            return self.set_row(item[0], value)
        if isinstance(item,slice):
            raise ValueError(str(item)+' not supported.')
        if isinstance(item,int):
            return self.set_row(item, value)
        if isinstance(item,tuple):
            if len(item)!=2:
                raise TypeError('Only 2D indexing supported.')
            item_i, item_j = item
            if isinstance(item_i, int) and isinstance(item_j, slice):
                if item_j==colon_slice:
                    return self.set_row(item_i, value)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, slice) and isinstance(item_j, int):
                if item_i==colon_slice:
                    return self.set_col(item_j, value)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, int) and isinstance(item_j, int):
                return self.set_element(item_i, item_j, value)
            raise TypeError('Only (i,:), (:,j), (i,j) indexing supported.')
        raise TypeError('Can only set items of type slice or tuple')

class DenseConnectionMatrix(ConnectionMatrix, numpy.ndarray):
    '''
    Dense connection matrix
    
    See documentation for :class:`ConnectionMatrix` for details on
    connection matrix types.

    Thi smatrix implements a dense connection matrix. It is just
    a numpy array. The ``get_row`` and ``get_col`` methods return
    :class:`DenseConnectionVector`` objects.
    '''
    def __new__(subtype, data, **kwds):
        return numpy.array(data, **kwds).view(subtype)

    def __init__(self, val, **kwds):
        self.rows = [DenseConnectionVector(numpy.ndarray.__getitem__(self, i)) for i in xrange(val.shape[0])]
        self.cols = [DenseConnectionVector(numpy.ndarray.__getitem__(self, (slice(None), i))) for i in xrange(val.shape[1])]
    
    def get_rows(self, rows):
        return [self.rows[i] for i in rows]
    
    def get_row(self, i):
        return self.rows[i]
    
    def get_col(self, i):
        return self.cols[i]
    
    def set_row(self, i, x):
        self[i] = todense(x)
    
    def set_col(self, i, x):
        self[:, i] = todense(x)
    
    def get_element(self, i, j):
        return self[i,j]
    
    def set_element(self, i, j, val):
        self[i,j] = val
    insert = set_element
    
    def remove(self, i, j):
        self[i, j] = 0

class SparseConnectionMatrix(ConnectionMatrix):
    '''
    Sparse connection matrix
    
    See documentation for :class:`ConnectionMatrix` for details on
    connection matrix types.
        
    This class implements a sparse matrix with a fixed number of nonzero
    entries. Row access is very fast, and if the ``column_access`` keyword
    is ``True`` then column access is also supported (but is not as fast
    as row access).
    
    The matrix should be initialised with a scipy sparse matrix.
    
    The ``get_row`` and ``get_col`` methods return
    :class:`SparseConnectionVector` objects. In addition to the
    usual slicing operations supported, ``M[:]=val`` is supported, where
    ``val`` must be a scalar or an array of length ``nnz``.
    
    Implementation details:
    
    The values are stored in an array ``alldata`` of length ``nnz`` (number
    of nonzero entries). The slice ``alldata[rowind[i]:rowind[i+1]]`` gives
    the values for row ``i``. These slices are stored in the list ``rowdata``
    so that ``rowdata[i]`` is the data for row ``i``. The array ``rowj[i]``
    gives the corresponding column ``j`` indices. For row access, the
    memory requirements are 12 bytes per entry (8 bytes for the float value,
    and 4 bytes for the column indices). The array ``allj`` of length ``nnz``
    gives the column ``j`` coordinates for each element in ``alldata`` (the
    elements of ``rowj`` are slices of this array so no extra memory is
    used).
    
    If column access is being used, then in addition to the above there are
    lists ``coli`` and ``coldataindices``. For column ``j``, the array
    ``coli[j]`` gives the row indices for the data values in column ``j``,
    while ``coldataindices[j]`` gives the indices in the array ``alldata``
    for the values in column ``j``. Column access therefore involves a
    copy operation rather than a slice operation. Column access increases
    the memory requirements to 20 bytes per entry (4 extra bytes for the
    row indices and 4 extra bytes for the data indices).
    '''
    def __init__(self, val, column_access=False):
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
        i = 0 # i points to the current index in the alldata array as we go through row by row
        for c in xrange(val.shape[0]):
            # extra the row values and column indices of row c of the initialising matrix
            # this works for any of the scipy sparse matrix formats
            if isinstance(val, scipy.sparse.lil_matrix):
                r = val.rows[c]
                d = val.data[c]
            else:
                sr = val[c, :]
                sr = sr.tolil()
                r = sr.rows[0]
                d = sr.data[0]
            # copy the values into the alldata array, the indices into the allj array, and
            # so forth
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
            # counts the number of nonzero elements in each column
            counts = numpy.histogram(allj, numpy.arange(val.shape[1]+1, dtype=int), new=True)[0]
            # now we have to go through one by one unfortunately, and so we keep curcdi, the
            # current column data index for each column
            curcdi = numpy.zeros(val.shape[1], dtype=int)
            # initialise the memory for the column data indices
            for j in xrange(val.shape[1]):
                coldataindices.append(numpy.zeros(counts[j], dtype=int))
            # one by one for every element, update the dataindices and curcdi data pointers
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
            self.coli = coli
            self.coldataindices = coldataindices
        self.rows = [SparseConnectionVector(self.shape[1], self.rowj[i], self.rowdata[i]) for i in xrange(self.shape[0])]
    
    def getnnz(self):
        return self.nnz
    
    def get_element(self, i, j):
        n = searchsorted(self.rowj[i], j)
        if n>=len(self.rowj[i]) or self.rowj[i][n]!=j:
            return 0
        return self.rowdata[i][n]
    
    def set_element(self, i, j, x):
        n = searchsorted(self.rowj[i], j)
        if n>=len(self.rowj[i]) or self.rowj[i][n]!=j:
            raise ValueError('Insertion of new elements not supported for SparseConnectionMatrix.')
        self.rowdata[i][n] = x
    
    def get_row(self, i):
        return self.rows[i]
    
    def get_rows(self, rows):
        return [self.rows[i] for i in rows]
    
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
            if isinstance(val, numpy.ndarray):
                val = asarray(val)
                self.rowdata[i][:] = val[self.rowj[i]]
            else:
                self.rowdata[i][:] = val
    
    def set_col(self, j, val):
        if self.column_access:
            if isinstance(val, SparseConnectionVector):
                if val.ind is not self.coli[j]:
                    if not (val.ind==self.coli[j]).all():
                        raise ValueError('Sparse col setting must use same indices.')
                self.alldata[self.coldataindices[j]] = val
            else:
                if isinstance(val, numpy.ndarray):
                    val = asarray(val)
                    self.alldata[self.coldataindices[j]] = val[self.coli[j]]
                else:
                    self.alldata[self.coldataindices[j]] = val
        else:
            raise TypeError('No column access.')
    
    def __setitem__(self, item, value):
        if item==colon_slice:
            self.alldata[:] = value
        else:
            ConnectionMatrix.__setitem__(self, item, value)

class DynamicConnectionMatrix(ConnectionMatrix):
    '''
    Dynamic (sparse) connection matrix
    
    See documentation for :class:`ConnectionMatrix` for details on
    connection matrix types.
        
    This class implements a sparse matrix with a variable number of nonzero
    entries. Row access and column access are provided, but are not as fast
    as for :class:`SparseConnectionMatrix`.
    
    The matrix should be initialised with a scipy sparse matrix.
    
    The ``get_row`` and ``get_col`` methods return
    :class:`SparseConnectionVector` objects. In addition to the
    usual slicing operations supported, ``M[:]=val`` is supported, where
    ``val`` must be a scalar or an array of length ``nnz``.
    
    Implementation details:
    
    The values are stored in an array ``alldata`` of length ``nnzmax`` (maximum
    number of nonzero entries). This is a dynamic array, see:
        http://en.wikipedia.org/wiki/Dynamic_array
    You can set the resizing constant with the argument ``dynamic_array_const``.
    Normally the default value 2 is fine but if memory is a worry it could be
    made smaller.
    
    Rows and column point in to this data array, and the list ``rowj`` consists
    of an array of column indices for each row, with ``coli`` containing arrays
    of row indices for each column. Similarly, ``rowdataind`` and ``coldataind``
    consist of arrays of pointers to the indices in the ``alldata`` array. 
    '''
    def __init__(self, val, nnzmax=None, dynamic_array_const=2, **kwds):
        self.shape = val.shape
        self.dynamic_array_const = dynamic_array_const
        if nnzmax is None or nnzmax<val.getnnz():
            nnzmax = val.getnnz()
        self.nnzmax = nnzmax
        self.nnz = val.getnnz()
        self.alldata = numpy.zeros(nnzmax)
        self.unusedinds = range(self.nnz, self.nnzmax)
        i = 0
        self.rowj = []
        self.rowdataind = []
        alli = zeros(self.nnz, dtype=int)
        allj = zeros(self.nnz, dtype=int)
        for c in xrange(val.shape[0]):
            # extra the row values and column indices of row c of the initialising matrix
            # this works for any of the scipy sparse matrix formats
            if isinstance(val, scipy.sparse.lil_matrix):
                r = val.rows[c]
                d = val.data[c]
            else:
                sr = val[c, :]
                sr = sr.tolil()
                r = sr.rows[0]
                d = sr.data[0]
            self.alldata[i:i+len(d)] = d
            self.rowj.append(array(r, dtype=int))
            self.rowdataind.append(arange(i, i+len(d)))
            allj[i:i+len(d)] = r
            alli[i:i+len(d)] = c
            i += len(d)
        # now update the coli and coldataind variables
        self.coli = []
        self.coldataind = []
        # counts the number of nonzero elements in each column
        counts = numpy.histogram(allj, numpy.arange(val.shape[1]+1, dtype=int), new=True)[0]
        # now we have to go through one by one unfortunately, and so we keep curcdi, the
        # current column data index for each column
        curcdi = numpy.zeros(val.shape[1], dtype=int)
        # initialise the memory for the column data indices
        for j in xrange(val.shape[1]):
            self.coldataind.append(numpy.zeros(counts[j], dtype=int))
        # one by one for every element, update the dataindices and curcdi data pointers
        for i, j in enumerate(allj):
            self.coldataind[j][curcdi[j]] = i
            curcdi[j]+=1
        for j in xrange(val.shape[1]):
            self.coli.append(alli[self.coldataind[j]])
    
    def getnnz(self):
        return self.nnz
    
    def insert(self, i, j, x):
        n = searchsorted(self.rowj[i], j)
        if n<len(self.rowj[i]) and self.rowj[i][n]==j:
            self.alldata[self.rowdataind[i][n]] = x
            return
        m = searchsorted(self.coli[j], i)        
        if self.nnz==self.nnzmax:
            # reallocate memory using a dynamic array structure (amortized O(1) cost for append)
            newnnzmax = int(self.nnzmax*self.dynamic_array_const)
            if newnnzmax<=self.nnzmax:
                newnnzmax += 1
            if newnnzmax>self.shape[0]*self.shape[1]:
                newnnzmax = self.shape[0]*self.shape[1]
            self.alldata = hstack((self.alldata, numpy.zeros(newnnzmax-self.nnzmax, dtype=self.alldata.dtype)))
            self.unusedinds.extend(range(self.nnz, newnnzmax))
            self.nnzmax = newnnzmax
        newind = self.unusedinds.pop(-1)
        self.alldata[newind] = x
        self.nnz += 1
        # update row
        newrowj = numpy.zeros(len(self.rowj[i])+1, dtype=int)
        newrowj[:n] = self.rowj[i][:n]
        newrowj[n] = j
        newrowj[n+1:] = self.rowj[i][n:]
        self.rowj[i] = newrowj
        newrowdataind = numpy.zeros(len(self.rowdataind[i])+1, dtype=int)
        newrowdataind[:n] = self.rowdataind[i][:n]
        newrowdataind[n] = newind
        newrowdataind[n+1:] = self.rowdataind[i][n:]
        self.rowdataind[i] = newrowdataind
        # update col
        newcoli = numpy.zeros(len(self.coli[j])+1, dtype=int)
        newcoli[:m] = self.coli[j][:m]
        newcoli[m] = i
        newcoli[m+1:] = self.coli[j][m:]
        self.coli[j] = newcoli
        newcoldataind = numpy.zeros(len(self.coldataind[j])+1, dtype=int)
        newcoldataind[:m] = self.coldataind[j][:m]
        newcoldataind[m] = newind
        newcoldataind[m+1:] = self.coldataind[j][m:]
        self.coldataind[j] = newcoldataind
    
    def remove(self, i, j):
        n = searchsorted(self.rowj[i], j)
        if n>=len(self.rowj[i]) or self.rowj[i][n]!=j:
            raise ValueError('No element to remove at position '+str(i,j))
        oldind = self.rowdataind[i][n]
        self.unusedinds.append(oldind)
        self.nnz -= 1
        m = searchsorted(self.coli[j], i)
        # update row
        newrowj = numpy.zeros(len(self.rowj[i])-1, dtype=int)
        newrowj[:n] = self.rowj[i][:n]
        newrowj[n:] = self.rowj[i][n+1:]
        self.rowj[i] = newrowj
        newrowdataind = numpy.zeros(len(self.rowdataind[i])-1, dtype=int)
        newrowdataind[:n] = self.rowdataind[i][:n]
        newrowdataind[n:] = self.rowdataind[i][n+1:]
        self.rowdataind[i] = newrowdataind
        # update col
        newcoli = numpy.zeros(len(self.coli[j])-1, dtype=int)
        newcoli[:m] = self.coli[j][:m]
        newcoli[m:] = self.coli[j][m+1:]
        self.coli[j] = newcoli
        newcoldataind = numpy.zeros(len(self.coldataind[j])-1, dtype=int)
        newcoldataind[:m] = self.coldataind[j][:m]
        newcoldataind[m:] = self.coldataind[j][m+1:]
        self.coldataind[j] = newcoldataind
    
    def get_row(self, i):
        return SparseConnectionVector(self.shape[1], self.rowj[i], self.alldata[self.rowdataind[i]])
    
    def get_rows(self, rows):
        return [SparseConnectionVector(self.shape[1], self.rowj[i], self.alldata[self.rowdataind[i]]) for i in rows]
    
    def get_col(self, j):
        return SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataind[j]])
    
    def set_row(self, i, val):
        if isinstance(val, SparseConnectionVector):
            if val.ind is not self.rowj[i]:
                if not (val.ind==self.rowj[i]).all():
                    raise ValueError('Sparse row setting must use same indices.')
            self.alldata[self.rowdataind[i]] = val
        else:
            if isinstance(val, numpy.ndarray):
                val = asarray(val)
                self.alldata[self.rowdataind[i]] = val[self.rowj[i]]
            else:
                self.alldata[self.rowdataind[i]] = val
    
    def set_col(self, j, val):
        if isinstance(val, SparseConnectionVector):
            if val.ind is not self.coli[j]:
                if not (val.ind==self.coli[j]).all():
                    raise ValueError('Sparse row setting must use same indices.')
            self.alldata[self.coldataind[j]] = val
        else:
            if isinstance(val, numpy.ndarray):
                val = asarray(val)
                self.alldata[self.coldataind[j]] = val[self.coli[j]]
            else:
                self.alldata[self.coldataind[j]] = val
    
    def __setitem__(self, item, value):
        if item==colon_slice:
            self.alldata[:self.nnz] = value
        else:
            ConnectionMatrix.__setitem__(self, item, value)

class Connection(Connection):
    @check_units(delay=second)
    def __init__(self,source,target,state=0,delay=0*msecond,modulation=None,
                 structure='sparse',**kwds):
        self.source = source # pointer to source group
        self.target = target # pointer to target group
        if isinstance(state, str): # named state variable
            self.nstate = target.get_var_index(state)
        else:
            self.nstate = state # target state index
        if isinstance(modulation, str): # named state variable
            self._nstate_mod = source.get_var_index(modulation)
        else:
            self._nstate_mod = modulation # source state index
        if isinstance(structure, str):
            structure = construction_matrix_register[structure]
        self.W = structure((len(source),len(target)),**kwds)
        self.iscompressed = False # True if compress() has been called
        source.set_max_delay(delay)
        self.delay = int(delay/source.clock.dt) # Synaptic delay in time bins
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        
    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
            sv=self.target._S[self.nstate]
            if self._nstate_mod is not None:
                sv_pre = self.source._S[self._nstate_mod]
            rows = self.W.get_rows(spikes)
            if isinstance(rows[0], SparseConnectionVector):
                if self._nstate_mod is None:
                    if self._useaccel:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyArrayObject* _row = convert_to_numpy(rowinds[j], "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyArrayObject* _data = convert_to_numpy(datas[j], "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    int m = row.numElements();
                                    for(int k=0;k<m;k++)
                                    {
                                        sv(row(k)) += data(k);
                                    }
                                }
                                """
                        weave.inline(code,['sv','rowinds','datas','spikes','nspikes'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        for row in rows:
                            sv[row.ind] += row
                else:
                    if self._useaccel:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyArrayObject* _row = convert_to_numpy(rowinds[j], "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyArrayObject* _data = convert_to_numpy(datas[j], "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    int m = row.numElements();
                                    for(int k=0;k<m;k++)
                                    {
                                        sv(row(k)) += data(k)*sv_pre(row(k));
                                    }
                                }
                                """
                        weave.inline(code,['sv','sv_pre','rowinds','datas','spikes','nspikes'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        for row in rows:
                            sv[row.ind] += row*sv_pre[row.ind]
            else:
                if self._nstate_mod is None:
                    if self._useaccel:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(sv)
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyArrayObject* _row = convert_to_numpy(rows[j], "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    for(int k=0;k<N;k++)
                                        sv(k) += row(k);
                                }
                                """
                        weave.inline(code,['sv','spikes','nspikes','N', 'rows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        for row in rows:
                            sv += row
                else:
                    if self._useaccel:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(sv)
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyArrayObject* _row = convert_to_numpy(rows[j], "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    for(int k=0;k<N;k++)
                                        sv(k) += row(k)*sv_pre(k);
                                }
                                """
                        weave.inline(code,['sv','sv_pre','spikes','nspikes','N', 'rows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        for row in rows:
                            sv += row*sv_pre
                    
    def compress(self):
        if not self.iscompressed:
            self.W = self.W.connection_matrix()
            self.iscompressed = True

if __name__=='__main__':
    
    # original version of Connection still slightly faster with compilation on, but not much...
    # on my laptop with new Connection:
    #  initialise:  1.3900001049
    #  run:        10.1719999313
    # and with old Connection:
    #  initialise:  1.70299983025 (slower surprisingly!)
    #  run:         9.18700003624 (about 10% faster, but this should be fixable?)
    #from brian import Connection
    
    #set_global_preferences(useweave=False)
    
    import time
    
    start = time.time()
    
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    
    P=NeuronGroup(4000,model=eqs,
                  threshold=-50*mV,reset=-60*mV,
                  refractory=5*ms)
    P.v=-60*mV+10*mV*rand(len(P))
    Pe=P.subgroup(3200)
    Pi=P.subgroup(800)
    
    Ce=Connection(Pe,P,'ge',structure='dynamic')
    Ci=Connection(Pi,P,'gi',structure='dynamic')
    
    Ce.connect_random(Pe, P, 0.02,weight=9*mV)
    Ci.connect_random(Pi, P, 0.02,weight=-9*mV)

    M = PopulationSpikeCounter(P)
    #M = SpikeMonitor(P)
    
    net = Network(P, Ce, Ci, M)
    
    net.run(1*ms)

    print time.time()-start
    start = time.time()
    
    def f():
        run(1*second)
    
    f()
    print time.time()-start

    print M.nspikes
    
    if hasattr(M, 'spikes') and len(M.spikes):
        raster_plot(M)
        show()

#    import cProfile as profile
#    import pstats
#    profile.run('f()','newconn.prof')
#    stats = pstats.Stats('newconn.prof')
#    #stats.strip_dirs()
#    stats.sort_stats('cumulative', 'calls')
#    stats.print_stats(50)
    
    #raster_plot(M)
    #show()    
#    x = scipy.sparse.lil_matrix((5,5))
#    x[2:4,1:3] = 1
#    print x.todense()
#    y = DynamicConnectionMatrix(x)
#    print y.alldata
#    print y.rowj
#    print y.rowdataind
#    print y.unusedinds
#    for i in range(5):
#        print y[i,:].todense()
#    print
#    for i in range(5):
#        print y[:,i].todense()
#    print
#    y.remove(2,1)
#    print y.todense()
#    print y.alldata
#    print y.rowj
#    print y.rowdataind
#    print y.unusedinds
#    y.insert(0,1,2)
#    print y.todense()
#    print y.alldata
#    print y.rowj
#    print y.rowdataind
#    print y.unusedinds
#    y.insert(0,1,3)
#    print y.todense()
#    print y.alldata
#    print y.rowj
#    print y.rowdataind
#    print y.unusedinds
#    y.insert(0,2,4)
#    print y.todense()
#    print y.alldata
#    print y.rowj
#    print y.rowdataind
#    print y.unusedinds
#    for i in range(5):
#        print y[:,i].todense()
#    print
#    y.insert(4,4,5)
#    print y.todense()
#    print y.alldata
#    print y.rowj
#    print y.rowdataind
#    print y.unusedinds
#    print
#    y[0,:] *= 2
#    print y.todense()
#    print
#    y[:,1] *= 3
#    print y.todense()
#    x = SparseConstructionMatrix((5,5))
#    x[2:4,1:3] = 1
#    print x.todense()
#    y = x.connection_matrix()
#    print y.alldata
#    print y.getnnz()
#    for i in range(5):
#        print y[i,:].todense()
#    print
#    for i in range(5):
#        print y[:,i].todense()
#    y[2,:]*=2#y[2,:]*2
#    y[:,2]*=3
#    print y.todense()
#    y[:]=7
#    print y.todense()
#    print y[2,1]
#    y[2,1]=8
#    print y.todense()
#    print y[0,0]
    
#    x = SparseConnectionVector(10, [1,2], [3.,4.])
#    z = SparseConnectionVector(10, [5,6], [7.,8.])
#    y = randn(10)
#    #print x*x
#    print (x+y)+x
#    print 2*asarray(x)+y[x.ind]
#    print x*y
#    print asarray(x)*y[x.ind]
#    #print x*z
#    import time
#    n = 2000
#    m = 1000
#    x = SparseConnectionVector(n, arange(m), arange(m, dtype=float))
#    y = randn(n)
#    z = asarray(x)
#    start = time.time()
#    for _ in xrange(10000):
#        x+y
#    print time.time()-start
#    start = time.time()
#    for _ in xrange(10000):
#        z+y[x.ind]
#    print time.time()-start
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