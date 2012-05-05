from base import *
from sparsematrix import *
from connectionvector import *
import gc

__all__ = [
         'ConnectionMatrix',
         'SparseConnectionMatrix',
         'DenseConnectionMatrix',
         'DynamicConnectionMatrix',
         'set_connection_from_sparse',
         ]

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
    ``get_cols(cols)``
        Returns a list of cols, should be implemented without Python
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

    def get_cols(self, cols):
        return [self.get_col(i) for i in cols]

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
        if isinstance(item, tuple) and isinstance(item[0], int) and item[1] == colon_slice:
            return self.get_row(item[0])
        if isinstance(item, slice):
            if item == colon_slice:
                return self
            else:
                raise ValueError(str(item) + ' not supported.')
        if isinstance(item, int):
            return self.get_row(item)
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError('Only 2D indexing supported.')
            item_i, item_j = item
            if isinstance(item_i, int) and isinstance(item_j, slice):
                if item_j == colon_slice:
                    return self.get_row(item_i)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, slice) and isinstance(item_j, int):
                if item_i == colon_slice:
                    return self.get_col(item_j)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, int) and isinstance(item_j, int):
                return self.get_element(item_i, item_j)
            raise TypeError('Only (i,:), (:,j), (i,j) indexing supported.')
        raise TypeError('Can only get items of type slice or tuple')

    def __setitem__(self, item, value):
        if isinstance(item, tuple) and isinstance(item[0], int) and item[1] == colon_slice:
            return self.set_row(item[0], value)
        if isinstance(item, slice):
            raise ValueError(str(item) + ' not supported.')
        if isinstance(item, int):
            return self.set_row(item, value)
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError('Only 2D indexing supported.')
            item_i, item_j = item
            if isinstance(item_i, int) and isinstance(item_j, slice):
                if item_j == colon_slice:
                    return self.set_row(item_i, value)
                raise ValueError('Only ":" indexing supported.')
            if isinstance(item_i, slice) and isinstance(item_j, int):
                if item_i == colon_slice:
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

    This matrix implements a dense connection matrix. It is just
    a numpy array. The ``get_row`` and ``get_col`` methods return
    :class:`DenseConnectionVector`` objects.
    '''
    def __new__(subtype, data, **kwds):
        if 'copy' not in kwds:
            kwds = dict(kwds.iteritems())
            kwds['copy'] = False
        return numpy.array(data, **kwds).view(subtype)

    def __init__(self, val, **kwds):
        # precompute rows and cols for fast returns by get_rows etc.
        self.rows = [DenseConnectionVector(numpy.ndarray.__getitem__(self, i)) for i in xrange(val.shape[0])]
        self.cols = [DenseConnectionVector(numpy.ndarray.__getitem__(self, (slice(None), i))) for i in xrange(val.shape[1])]

    def get_rows(self, rows):
        return [self.rows[i] for i in rows]

    def get_cols(self, cols):
        return [self.cols[i] for i in cols]

    def get_row(self, i):
        return self.rows[i]

    def get_col(self, i):
        return self.cols[i]

    def set_row(self, i, x):
        numpy.ndarray.__setitem__(self, i, todense(x))

    def set_col(self, i, x):
        numpy.ndarray.__setitem__(self, (colon_slice, i), todense(x))
        #self[:, i] = todense(x)

    def get_element(self, i, j):
        return numpy.ndarray.__getitem__(self, (i, j))
        #return self[i,j]

    def set_element(self, i, j, val):
        numpy.ndarray.__setitem__(self, (i, j), val)
        #self[i,j] = val
    insert = set_element

    def remove(self, i, j):
        numpy.ndarray.__setitem__(self, (i, j), 0)
        #self[i, j] = 0


class SparseConnectionMatrix(ConnectionMatrix):
    '''
    Sparse connection matrix
    
    See documentation for :class:`ConnectionMatrix` for details on
    connection matrix types.
        
    This class implements a sparse matrix with a fixed number of nonzero
    entries. Row access is very fast, and if the ``column_access`` keyword
    is ``True`` then column access is also supported (but is not as fast
    as row access). If the ``use_minimal_indices`` keyword is ``True`` then
    the neuron and synapse indices will use the smallest possible integer
    type (16 bits for neuron indices if the number of neurons is less than
    ``2**16``, otherwise 32 bits). Otherwise, it will use the word size for the
    CPU architecture (32 or 64 bits).
    
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
    
    TODO: update size numbers when use_minimal_indices=True for different
    architectures.
    '''
    def __init__(self, val, column_access=True, use_minimal_indices=False, **kwds):
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        self.nnz = nnz = val.getnnz()# nnz stands for number of nonzero entries
        alldata = numpy.zeros(nnz)
        self.neuron_index_dtype = int
        self.synapse_index_dtype = int
        if use_minimal_indices:
            if max(val.shape)<2**16:
                self.neuron_index_dtype = uint16
            else:
                self.neuron_index_dtype = uint32
            self.synapse_index_dtype = uint32
        if column_access:
            colind = numpy.zeros(val.shape[1] + 1,
                                 dtype=self.synapse_index_dtype)
        allj = numpy.zeros(nnz,
                           dtype=self.neuron_index_dtype)
        rowind = numpy.zeros(val.shape[0] + 1,
                             dtype=self.synapse_index_dtype)
        rowdata = []
        rowj = []
        if column_access:
            coli = []
            coldataindices = []
        i = 0 # i points to the current index in the alldata array as we go through row by row
        for c in xrange(val.shape[0]):
            # extra the row values and column indices of row c of the initialising matrix
            # this works for any of the scipy sparse matrix formats
            if isinstance(val, sparse.lil_matrix):
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
            alldata[i:i + len(d)] = d
            allj[i:i + len(r)] = r
            rowdata.append(alldata[i:i + len(d)])
            rowj.append(allj[i:i + len(r)])
            i = i + len(r)
        rowind[val.shape[0]] = i
        if column_access:
            # counts the number of nonzero elements in each column
            counts = zeros(val.shape[1],
                           dtype=self.synapse_index_dtype)
            if len(allj):
                bincounts = numpy.bincount(allj)
            else:
                bincounts = numpy.array([], dtype=int)
            counts[:len(bincounts)] = bincounts # ensure that counts is the right length
            # two algorithms depending on whether weave is available
            if self._useaccel:
                # this algorithm just goes through one by one adding each
                # element to the appropriate bin whose sizes we have
                # precomputed. alldi will contain all the data indices
                # in blocks alldi[s[i]:s[i+1]] of length counts[i], and
                # curcdi[i] is the current offset into each block. s is
                # therefore just the cumulative sum of counts.
                curcdi = numpy.zeros(val.shape[1], dtype=int)
                allcoldataindices = numpy.zeros(nnz,
                                                dtype=self.synapse_index_dtype)
                colind[:] = numpy.hstack(([0], cumsum(counts)))
                colalli = numpy.zeros(nnz,
                                      dtype=self.neuron_index_dtype)
                numrows = val.shape[0]
                code = '''
                int i = 0;
                for(int k=0;k<nnz;k++)
                {
                    while(k>=rowind[i+1]) i++;
                    int j = allj[k];
                    allcoldataindices[colind[j]+curcdi[j]] = k;
                    colalli[colind[j]+curcdi[j]] = i;
                    curcdi[j]++;
                }
                '''
                weave.inline(code, ['nnz', 'allj', 'allcoldataindices',
                                    'rowind', 'numrows',
                                    'curcdi', 'colind', 'colalli'],
                             compiler=self._cpp_compiler,
                             extra_compile_args=self._extra_compile_args,
                             )
                # now store the blocks of allcoldataindices in coldataindices and update coli too
                for i in xrange(len(colind) - 1):
                    D = allcoldataindices[colind[i]:colind[i + 1]]
                    I = colalli[colind[i]:colind[i + 1]]
                    coldataindices.append(D)
                    coli.append(I)
            else:
                # now allj[a] will be the columns in order, so that
                # the first counts[0] elements of allj[a] will be 0,
                # or in other words the first counts[0] elements of a
                # will be the data indices of the elements (i,j) with j==0
                # mergesort is necessary because we want the relative ordering
                # of the elements of a within a block to be maintained
                allcoldataindices = a = argsort(allj, kind='mergesort')
                # this defines colind so that a[colind[i]:colind[i+1]] are the data
                # indices where j==i
                colind[:] = numpy.hstack(([0], cumsum(counts)))
                # this computes the row index of each entry by first generating
                # expanded_row_indices which gives the corresponding row index
                # for each entry enumerated row-by-row, and then using the
                # array allcoldataindices to index this array to convert into
                # the corresponding row index for each entry enumerated
                # col-by-col.
                if len(a):
                    expanded_row_indices = empty(len(a),
                                                 dtype=self.neuron_index_dtype)
                    for k, (i, j) in enumerate(zip(rowind[:-1], rowind[1:])):
                        expanded_row_indices[i:j] = k
                    colalli = expanded_row_indices[a]
                else:
                    colalli = numpy.zeros(nnz,
                                          dtype=self.neuron_index_dtype)
                # in this loop, I are the data indices where j==i
                # and alli[I} are the corresponding i coordinates
                for i in xrange(len(colind) - 1):
                    D = a[colind[i]:colind[i + 1]]
                    I = colalli[colind[i]:colind[i + 1]]
                    coldataindices.append(D)
                    coli.append(I)

        self.alldata = alldata
        self.rowdata = rowdata
        self.allj = allj
        self.rowj = rowj
        self.rowind = rowind
        self.shape = val.shape
        self.column_access = column_access
        if column_access:
            self.colalli = colalli
            self.coli = coli
            self.coldataindices = coldataindices
            self.allcoldataindices = allcoldataindices
            self.colind = colind
        self.rows = [SparseConnectionVector(self.shape[1], self.rowj[i], self.rowdata[i]) for i in xrange(self.shape[0])]

    def getnnz(self):
        return self.nnz

    def get_element(self, i, j):
        n = searchsorted(self.rowj[i], j)
        if n >= len(self.rowj[i]) or self.rowj[i][n] != j:
            return 0
        return self.rowdata[i][n]

    def set_element(self, i, j, x):
        n = searchsorted(self.rowj[i], j)
        if n >= len(self.rowj[i]) or self.rowj[i][n] != j:
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

    def get_cols(self, cols):
        if self.column_access:
            return [SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataindices[j]]) for j in cols]
        else:
            raise TypeError('No column access.')

    def set_row(self, i, val):
        if isinstance(val, SparseConnectionVector):
            if val.ind is not self.rowj[i]:
                if not equal(val.ind, self.rowj[i]).all():
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
                    if not (val.ind == self.coli[j]).all():
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
        if item == colon_slice:
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
    
    **Implementation details**
    
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
        if nnzmax is None or nnzmax < val.getnnz():
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
            if isinstance(val, sparse.lil_matrix):
                r = val.rows[c]
                d = val.data[c]
            else:
                sr = val[c, :]
                sr = sr.tolil()
                r = sr.rows[0]
                d = sr.data[0]
            self.alldata[i:i + len(d)] = d
            self.rowj.append(array(r, dtype=int))
            self.rowdataind.append(arange(i, i + len(d)))
            allj[i:i + len(d)] = r
            alli[i:i + len(d)] = c
            i += len(d)
        # now update the coli and coldataind variables
        self.coli = []
        self.coldataind = []
        # counts the number of nonzero elements in each column
        if numpy.__version__ >= '1.3.0':
            counts = numpy.histogram(allj, numpy.arange(val.shape[1] + 1, dtype=int))[0]
        else:
            counts = numpy.histogram(allj, numpy.arange(val.shape[1] + 1, dtype=int), new=True)[0]
        # now we have to go through one by one unfortunately, and so we keep curcdi, the
        # current column data index for each column
        curcdi = numpy.zeros(val.shape[1], dtype=int)
        # initialise the memory for the column data indices
        for j in xrange(val.shape[1]):
            self.coldataind.append(numpy.zeros(counts[j], dtype=int))
        # one by one for every element, update the dataindices and curcdi data pointers
        for i, j in enumerate(allj):
            self.coldataind[j][curcdi[j]] = i
            curcdi[j] += 1
        for j in xrange(val.shape[1]):
            self.coli.append(alli[self.coldataind[j]])

    def getnnz(self):
        return self.nnz

    def insert(self, i, j, x):
        n = searchsorted(self.rowj[i], j)
        if n < len(self.rowj[i]) and self.rowj[i][n] == j:
            self.alldata[self.rowdataind[i][n]] = x
            return
        m = searchsorted(self.coli[j], i)
        if self.nnz == self.nnzmax:
            # reallocate memory using a dynamic array structure (amortized O(1) cost for append)
            newnnzmax = int(self.nnzmax * self.dynamic_array_const)
            if newnnzmax <= self.nnzmax:
                newnnzmax += 1
            if newnnzmax > self.shape[0] * self.shape[1]:
                newnnzmax = self.shape[0] * self.shape[1]
            self.alldata = hstack((self.alldata, numpy.zeros(newnnzmax - self.nnzmax, dtype=self.alldata.dtype)))
            self.unusedinds.extend(range(self.nnz, newnnzmax))
            self.nnzmax = newnnzmax
        newind = self.unusedinds.pop(-1)
        self.alldata[newind] = x
        self.nnz += 1
        # update row
        newrowj = numpy.zeros(len(self.rowj[i]) + 1, dtype=int)
        newrowj[:n] = self.rowj[i][:n]
        newrowj[n] = j
        newrowj[n + 1:] = self.rowj[i][n:]
        self.rowj[i] = newrowj
        newrowdataind = numpy.zeros(len(self.rowdataind[i]) + 1, dtype=int)
        newrowdataind[:n] = self.rowdataind[i][:n]
        newrowdataind[n] = newind
        newrowdataind[n + 1:] = self.rowdataind[i][n:]
        self.rowdataind[i] = newrowdataind
        # update col
        newcoli = numpy.zeros(len(self.coli[j]) + 1, dtype=int)
        newcoli[:m] = self.coli[j][:m]
        newcoli[m] = i
        newcoli[m + 1:] = self.coli[j][m:]
        self.coli[j] = newcoli
        newcoldataind = numpy.zeros(len(self.coldataind[j]) + 1, dtype=int)
        newcoldataind[:m] = self.coldataind[j][:m]
        newcoldataind[m] = newind
        newcoldataind[m + 1:] = self.coldataind[j][m:]
        self.coldataind[j] = newcoldataind

    def remove(self, i, j):
        n = searchsorted(self.rowj[i], j)
        if n >= len(self.rowj[i]) or self.rowj[i][n] != j:
            raise ValueError('No element to remove at position ' + str(i, j))
        oldind = self.rowdataind[i][n]
        self.unusedinds.append(oldind)
        self.nnz -= 1
        m = searchsorted(self.coli[j], i)
        # update row
        newrowj = numpy.zeros(len(self.rowj[i]) - 1, dtype=int)
        newrowj[:n] = self.rowj[i][:n]
        newrowj[n:] = self.rowj[i][n + 1:]
        self.rowj[i] = newrowj
        newrowdataind = numpy.zeros(len(self.rowdataind[i]) - 1, dtype=int)
        newrowdataind[:n] = self.rowdataind[i][:n]
        newrowdataind[n:] = self.rowdataind[i][n + 1:]
        self.rowdataind[i] = newrowdataind
        # update col
        newcoli = numpy.zeros(len(self.coli[j]) - 1, dtype=int)
        newcoli[:m] = self.coli[j][:m]
        newcoli[m:] = self.coli[j][m + 1:]
        self.coli[j] = newcoli
        newcoldataind = numpy.zeros(len(self.coldataind[j]) - 1, dtype=int)
        newcoldataind[:m] = self.coldataind[j][:m]
        newcoldataind[m:] = self.coldataind[j][m + 1:]
        self.coldataind[j] = newcoldataind

    def get_element(self, i, j):
        n = searchsorted(self.rowj[i], j)
        if n >= len(self.rowj[i]) or self.rowj[i][n] != j:
            return 0
        return self.alldata[self.rowdataind[i][n]]

    set_element = insert

    def get_row(self, i):
        return SparseConnectionVector(self.shape[1], self.rowj[i], self.alldata[self.rowdataind[i]])

    def get_rows(self, rows):
        return [SparseConnectionVector(self.shape[1], self.rowj[i], self.alldata[self.rowdataind[i]]) for i in rows]

    def get_col(self, j):
        return SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataind[j]])

    def get_cols(self, cols):
        return [SparseConnectionVector(self.shape[0], self.coli[j], self.alldata[self.coldataind[j]]) for j in cols]

    def set_row(self, i, val):
        if isinstance(val, SparseConnectionVector):
            if val.ind is not self.rowj[i]:
                if not (val.ind == self.rowj[i]).all():
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
                if not (val.ind == self.coli[j]).all():
                    raise ValueError('Sparse row setting must use same indices.')
            self.alldata[self.coldataind[j]] = val
        else:
            if isinstance(val, numpy.ndarray):
                val = asarray(val)
                self.alldata[self.coldataind[j]] = val[self.coli[j]]
            else:
                self.alldata[self.coldataind[j]] = val

    def __setitem__(self, item, value):
        if item == colon_slice:
            self.alldata[:self.nnz] = value
        else:
            ConnectionMatrix.__setitem__(self, item, value)



class UnconstructedMatrix(object):
    pass

def make_sparse_connection_matrix(x, column_access=True):
    x = x.tocsr()
    if not x.has_sorted_indices:
        x.sort_indices()
    y = UnconstructedMatrix()
    y.__class__ = SparseConnectionMatrix
    y._useaccel = get_global_preference('useweave')
    y._cpp_compiler = get_global_preference('weavecompiler')
    y._extra_compile_args = ['-O3']
    if y._cpp_compiler == 'gcc':
        y._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
    y.nnz = nnz = int(x.getnnz())# nnz stands for number of nonzero entries
    y.alldata = alldata = x.data
    y.rowind = rowind = array(x.indptr, dtype=int, copy=False)
    y.allj = allj = array(x.indices, dtype=int, copy=False)
    if column_access:
        colind = numpy.zeros(x.shape[1]+1, dtype=int)
        coli = []
        coldataindices = []
    rowdata = []
    rowj = []
    i = 0 # i points to the current index in the alldata array as we go through row by row
    for c in xrange(x.shape[0]):
        k = y.rowind[c+1]-y.rowind[c]
        rowdata.append(y.alldata[i:i+k])
        rowj.append(y.allj[i:i+k])
        i += k
    if column_access:
        # counts the number of nonzero elements in each column
        counts = zeros(x.shape[1], dtype=int)
        if len(allj):
            bincounts = numpy.bincount(allj)
        else:
            bincounts = numpy.array([], dtype=int)
        counts[:len(bincounts)] = bincounts # ensure that counts is the right length
        # two algorithms depending on whether weave is available
        if y._useaccel:
            # this algorithm just goes through one by one adding each
            # element to the appropriate bin whose sizes we have
            # precomputed. alldi will contain all the data indices
            # in blocks alldi[s[i]:s[i+1]] of length counts[i], and
            # curcdi[i] is the current offset into each block. s is
            # therefore just the cumulative sum of counts.
            curcdi = numpy.zeros(x.shape[1], dtype=int)
            allcoldataindices = numpy.zeros(nnz, dtype=int)
            colind[:] = numpy.hstack(([0], cumsum(counts)))
            colalli = numpy.zeros(nnz, dtype=int)
            numrows = x.shape[0]
            code = '''
            int i = 0;
            for(int k=0;k<nnz;k++)
            {
                while(k>=rowind[i+1]) i++;
                int j = allj[k];
                allcoldataindices[colind[j]+curcdi[j]] = k;
                colalli[colind[j]+curcdi[j]] = i;
                curcdi[j]++;
            }
            '''
            weave.inline(code, ['nnz', 'allj', 'allcoldataindices',
                                'rowind', 'numrows',
                                'curcdi', 'colind', 'colalli'],
                         compiler=y._cpp_compiler,
                         extra_compile_args=y._extra_compile_args,
                         )
            # now store the blocks of allcoldataindices in coldataindices and update coli too
            for i in xrange(len(colind) - 1):
                D = allcoldataindices[colind[i]:colind[i + 1]]
                I = colalli[colind[i]:colind[i + 1]]
                coldataindices.append(D)
                coli.append(I)
        else:
            # now allj[a] will be the columns in order, so that
            # the first counts[0] elements of allj[a] will be 0,
            # or in other words the first counts[0] elements of a
            # will be the data indices of the elements (i,j) with j==0
            # mergesort is necessary because we want the relative ordering
            # of the elements of a within a block to be maintained
            allcoldataindices = a = argsort(allj, kind='mergesort')
            # this defines colind so that a[colind[i]:colind[i+1]] are the data
            # indices where j==i
            colind[:] = numpy.hstack(([0], cumsum(counts)))
            # this computes the row index of each entry by first generating
            # expanded_row_indices which gives the corresponding row index
            # for each entry enumerated row-by-row, and then using the
            # array allcoldataindices to index this array to convert into
            # the corresponding row index for each entry enumerated
            # col-by-col.
            if len(a):
                expanded_row_indices = empty(len(a), dtype=int)
                for k, (i, j) in enumerate(zip(rowind[:-1], rowind[1:])):
                    expanded_row_indices[i:j] = k
                colalli = expanded_row_indices[a]
            else:
                colalli = numpy.zeros(nnz, dtype=int)
            # in this loop, I are the data indices where j==i
            # and alli[I} are the corresponding i coordinates
            for i in xrange(len(colind) - 1):
                D = a[colind[i]:colind[i + 1]]
                I = colalli[colind[i]:colind[i + 1]]
                coldataindices.append(D)
                coli.append(I)

    y.rowdata = rowdata
    y.rowj = rowj
    y.shape = x.shape
    y.column_access = column_access
    if column_access:
        y.colalli = colalli
        y.coli = coli
        y.coldataindices = coldataindices
        y.allcoldataindices = allcoldataindices
        y.colind = colind
    y.rows = [SparseConnectionVector(y.shape[1], y.rowj[i], y.rowdata[i]) for i in xrange(y.shape[0])]
    return y

def set_connection_from_sparse(C, W, delay=None, column_access=True):
    C.W = make_sparse_connection_matrix(W, column_access=column_access)
    if delay is not None:
        C.delay = make_sparse_connection_matrix(delay, column_access=column_access)
    C.iscompressed = True
    gc.collect()
