# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
__all__=[
         'Connection', 'IdentityConnection', 'MultiConnection',
         'DelayConnection',
         #'HeterogeneousDelayConnection', # this class is not defined
         'random_matrix_fixed_column',
         'random_matrix',
#         'UserComputedConnectionMatrix',
#         'UserComputedSparseConnectionMatrix',
#         'random_row_func',
#         'random_sparse_row_func'
         'ConstructionMatrix', 'SparseConstructionMatrix', 'DenseConstructionMatrix', 'DynamicConstructionMatrix',
         'ConnectionMatrix', 'SparseConnectionMatrix', 'DenseConnectionMatrix', 'DynamicConnectionMatrix',
         'ConnectionVector', 'SparseConnectionVector', 'DenseConnectionVector',
         ]

import copy
from itertools import izip
import itertools
from random import sample
import bisect
from units import *
import types
import magic
from log import *
from numpy import *
from scipy import sparse,stats,rand,weave,linalg
import scipy
import numpy
import random as pyrandom
from scipy import random as scirandom
from utils.approximatecomparisons import is_within_absolute_tolerance
from globalprefs import *
from base import *
from stdunits import ms
# We should do this:
# from network import network_operation
# but instead we do it at the bottom of the module (see comments there for explanation)
    
effective_zero = 1e-40

def random_row_func(N, p, weight=1., initseed=None):
    '''
    Returns a random connectivity ``row_func`` for use with :class:`UserComputedConnectionMatrix`
    
    Gives equivalent output to the :meth:`Connection.connect_random` method.
    
    Arguments:
    
    ``N``
        The number of target neurons.
    ``p``
        The probability of a synapse.
    ``weight``
        The connection weight (must be a single value).
    ``initseed``
        The initial seed value (for reproducible results).
    '''
    if initseed is None:
        initseed = pyrandom.randint(100000,1000000) # replace this
    cur_row = numpy.zeros(N)
    myrange = numpy.arange(N, dtype=int)
    
    def row_func(i):
        pyrandom.seed(initseed+int(i))
        scirandom.seed(initseed+int(i))
        k = scirandom.binomial(N, p, 1)[0]
        cur_row[:] = 0.0
        cur_row[pyrandom.sample(myrange,k)] = weight
        return cur_row
    
    return row_func

#class UserComputedConnectionMatrix(ConnectionMatrix):
#    '''
#    A computed connection matrix defined by a user-specified function
#    
#    Normally this matrix will be initialised by passing the class
#    object to the :class:`Connection` object. In the initialisation
#    of the :class:`Connection` specify ``structure=UserComputedConnectionMatrix``
#    and add the keyword ``row_func=...``, e.g.::
#    
#        def f(i):
#            return max_weight*ones(N)/(1+(arange(N)-i)**2)
#        C = Connection(G1, G2, structure=UserComputedConnectionMatrix, row_func=f)
#    
#    Initialisation arguments:
#    
#    ``dims``
#        The pair ``(N,M)`` specifying the dimensions of the matrix.
#    ``row_func``
#        The function ``f(i)`` which returns an array of length ``M``,
#        the weight matrix for row ``i``. Note that you are responsible
#        for making sure the function returns consistent results (so
#        random functions should be initialised with a seed based on
#        the row ``i``).
#    
#    **Limitations**
#    
#    This type of connection matrix cannot be changed during a run, and
#    cannot be used with methods like :class:`Connection.connect_random`.
#    
#    **Efficiency considerations**
#    
#    This connection matrix is for dense connectivity, if the connectivity
#    is sparse you might get better performance with :class:`UserComputedSparseConnectionMatrix`.
#    '''
#    def __init__(self, dims, row_func):
#        self.sourcelen, self.targetlen = dims
#        self.row_func = row_func
#        
#    def get_row(self, i):
#        return self.row_func(i)
#    
#    def __getitem__(self, item):
#        if isinstance(item,int):
#            return self.get_row(item)
#        if isinstance(item,tuple):
#            if len(item)==2:
#                item_i, item_j = item
#                if isinstance(item_i, int) and isinstance(item_j, slice):
#                    if is_colon_slice(item_j):
#                        return self.get_row(item_i)
#        raise ValueError('Only "i,:" indexing supported.')
#
#def random_sparse_row_func(N, p, weight=1., initseed=None):
#    '''
#    Returns a random connectivity ``row_func`` for use with :class:`UserComputedSparseConnectionMatrix`
#    
#    Gives equivalent output to the :meth:`Connection.connect_random` method.
#    
#    Arguments:
#    
#    ``N``
#        The number of target neurons.
#    ``p``
#        The probability of a synapse.
#    ``weight``
#        The connection weight (must be a single value).
#    ``initseed``
#        The initial seed value (for reproducible results).
#    '''
#    if initseed is None:
#        initseed = pyrandom.randint(100000,1000000) # replace this
#    myrange = numpy.arange(N, dtype=int)
#    def row_func(i):
#        pyrandom.seed(initseed+int(i))
#        scirandom.seed(initseed+int(i))
#        k = scirandom.binomial(N, p, 1)[0]
#        return (pyrandom.sample(myrange,k), weight)
#    return row_func
#
#class UserComputedSparseConnectionMatrix(ConnectionMatrix):
#    '''
#    A computed sparse connection matrix defined by a user-specified function
#    
#    Normally this matrix will be initialised by passing the class
#    object to the :class:`Connection` object. In the initialisation
#    of the :class:`Connection` specify ``structure=UserComputedSparseConnectionMatrix``
#    and add the keyword ``row_func=...``, e.g.::
#    
#        def f(i):
#            if 0<i<N-1:
#                return ([i-1,i+1], weight*ones(2))
#            elif i>0:
#                return ([i-1], weight*ones(1))
#            else:
#                return ([i+1], weight*ones(1))
#        C = Connection(G1, G2, structure=UserComputedSparseConnectionMatrix, row_func=f)
#    
#    Initialisation arguments:
#    
#    ``dims``
#        The pair ``(N,M)`` specifying the dimensions of the matrix.
#    ``row_func``
#        The function ``f(i)`` which for a row ``i`` returns a pair ``(indices, values))``
#        consisting of a list or array ``indices`` with the indices of the
#        nonzero elements of the row, and an array of the same length ``values``
#        giving the weight matrix for those indices. Note that you are responsible
#        for making sure the function returns consistent results (so
#        random functions should be initialised with a seed based on
#        the row ``i``).
#    
#    **Limitations**
#    
#    This type of connection matrix cannot be changed during a run, and
#    cannot be used with methods like :class:`Connection.connect_random`.
#    
#    **Efficiency considerations**
#    
#    This connection matrix is for sparse connectivity, if the connectivity
#    is dense you might get better performance with :class:`UserComputedConnectionMatrix`.
#    '''
#    def __init__(self, dims, row_func):
#        self.sourcelen, self.targetlen = dims
#        self.row_func = row_func
#        self.cur_row = numpy.zeros(dims[1])
#        
#    def add_row(self,i,X):
#        indices, values = self.row_func(i)
#        X[indices]+=values
#        
#    def add_scaled_row(self,i,X,factor):
#        # modulation may not work? need factor[self.rows[i]] here? is factor a number or an array?
#        X[indices]+=factor*values
#        
#    def get_row(self, i):
#        indices, values = self.row_func(i)
#        self.cur_row[:] = 0.0
#        self.cur_row[indices] = values
#        return self.cur_row
#    
#    def __getitem__(self, item):
#        if isinstance(item,int):
#            return self.get_row(item)
#        if isinstance(item,tuple):
#            if len(item)==2:
#                item_i, item_j = item
#                if isinstance(item_i, int) and isinstance(item_j, slice):
#                    if is_colon_slice(item_j):
#                        return self.get_row(item_i)
#        raise ValueError('Only "i,:" indexing supported.')

    
colon_slice = slice(None,None,None)

def todense(x):
    if hasattr(x, 'todense'):
        return x.todense()
    return array(x, copy=False)

class ConnectionVector(object):
    '''
    Base class for connection vectors, just used for defining the interface
    
    ConnectionVector objects are returned by ConnectionMatrix objects when
    they retrieve rows or columns. At the moment, there are two choices,
    sparse or dense.
    
    This class has no real function at the moment.
    '''
    def todense(self):
        return NotImplemented
    
    def tosparse(self):
        return NotImplemented

class DenseConnectionVector(ConnectionVector, numpy.ndarray):
    '''
    Just a numpy array.
    '''
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
    '''
    Dense construction matrix. Essentially just numpy.ndarray.
    
    The ``connection_matrix`` method returns a :class:`DenseConnectionMatrix`
    object.
    
    The ``__setitem__`` method is overloaded so that you can set values with
    a sparse matrix.
    '''
    def __init__(self, val, **kwds):
        self.init_kwds = kwds
    
    def connection_matrix(self):
        kwds = self.init_kwds
        return DenseConnectionMatrix(self, **kwds)
    
    def __setitem__(self, index, W):
        # Make it work for sparse matrices
        if isinstance(W,scipy.sparse.spmatrix):
            ndarray.__setitem__(self,index,W.todense())
        else:
            ndarray.__setitem__(self,index,W)

oldscipy = scipy.__version__.startswith('0.6.')

class SparseMatrix(scipy.sparse.lil_matrix):
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
        TODO: once we've got this working in all cases, should we submit to scipy?
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
        elif oldscipy and isinstance(i, int) and isinstance(j, (list, tuple, numpy.ndarray)):
            row = dict(izip(self.rows[i], self.data[i]))
            try:
                row.update(dict(izip(j,W)))
            except TypeError:
                row.update(dict(izip(j,itertools.repeat(W))))
            items = row.items()
            items.sort()
            row, data = izip(*items)
            self.rows[i] = list(row)
            self.data[i] = list(data)
        else:
            scipy.sparse.lil_matrix.__setitem__(self,index,W)

class SparseConstructionMatrix(ConstructionMatrix, SparseMatrix):
    '''
    SparseConstructionMatrix is converted to SparseConnectionMatrix.
    '''
    def __init__(self, arg, **kwds):
        SparseMatrix.__init__(self, arg)
        self.init_kwds = kwds
        
    def connection_matrix(self):
        return SparseConnectionMatrix(self, **self.init_kwds)
            
class DynamicConstructionMatrix(ConstructionMatrix, SparseMatrix):
    '''
    DynamicConstructionMatrix is converted to DynamicConnectionMatrix.
    '''
    def __init__(self, arg, **kwds):
        SparseMatrix.__init__(self, arg)
        self.init_kwds = kwds
        
    def connection_matrix(self):
        return DynamicConnectionMatrix(self, **self.init_kwds)

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

    This matrix implements a dense connection matrix. It is just
    a numpy array. The ``get_row`` and ``get_col`` methods return
    :class:`DenseConnectionVector`` objects.
    '''
    def __new__(subtype, data, **kwds):
        return numpy.array(data, **kwds).view(subtype)

    def __init__(self, val, **kwds):
        # precompute rows and cols for fast returns by get_rows etc.
        self.rows = [DenseConnectionVector(numpy.ndarray.__getitem__(self, i)) for i in xrange(val.shape[0])]
        self.cols = [DenseConnectionVector(numpy.ndarray.__getitem__(self, (slice(None), i))) for i in xrange(val.shape[1])]
    
    def get_rows(self, rows):
        return [self.rows[i] for i in rows]
    
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
    def __init__(self, val, column_access=True):
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self.nnz = nnz = val.getnnz()# nnz stands for number of nonzero entries
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
            counts = zeros(val.shape[1], dtype=int)
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
                alldi = numpy.zeros(sum(counts), dtype=int)
                s = numpy.hstack(([0], cumsum(counts)))
                code = '''
                for(int i=0;i<nnz;i++)
                {
                    int j = allj[i];
                    alldi[s[j]+curcdi[j]] = i;
                    curcdi[j]++;
                }
                '''
                weave.inline(code, ['nnz', 'allj', 'alldi', 'curcdi', 's'],
                             compiler=self._cpp_compiler,
                             extra_compile_args=['-O3'])
                # now store the blocks of alldi in coldataindices and update coli too
                for i in xrange(len(s)-1):
                    I = alldi[s[i]:s[i+1]]
                    coldataindices.append(I)
                    coli.append(alli[I])
            else:
                # now allj[a] will be the columns in order, so that
                # the first counts[0] elements of allj[a] will be 0,
                # or in other words the first counts[0] elements of a
                # will be the data indices of the elements (i,j) with j==0
                # mergesort is necessary because we want the relative ordering
                # of the elements of a within a block to be maintained
                self.allj_sorted_indices = a = argsort(allj, kind='mergesort')
                # this defines s so that a[s[i]:s[i+1]] are the data
                # indices where j==i
                s = numpy.hstack(([0], cumsum(counts)))
                # in this loop, I are the data indices where j==i
                # and alli[I} are the corresponding i coordinates
                for i in xrange(len(s)-1):
                    I = a[s[i]:s[i+1]]
                    coldataindices.append(I)
                    coli.append(alli[I])

            # Pure Python version (slow in practice although O(nnz) in principle)
#            # now we have to go through one by one unfortunately, and so we keep curcdi, the
#            # current column data index for each column
#            curcdi = numpy.zeros(val.shape[1], dtype=int)
#            # initialise the memory for the column data indices
#            for j in xrange(val.shape[1]):
#                coldataindices.append(numpy.zeros(counts[j], dtype=int))
#            # one by one for every element, update the dataindices and curcdi data pointers
#            for i, j in enumerate(allj):
#                coldataindices[j][curcdi[j]] = i
#                curcdi[j]+=1
#            for j in xrange(val.shape[1]):
#                coli.append(alli[coldataindices[j]])
            
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

    def get_element(self, i, j):
        n = searchsorted(self.rowj[i], j)
        if n>=len(self.rowj[i]) or self.rowj[i][n]!=j:
            return 0
        return self.alldata[self.rowdataind[i][n]]

    set_element = insert
    
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

class Connection(magic.InstanceTracker, ObjectContainer):
    '''
    Mechanism for propagating spikes from one group to another

    A Connection object declares that when spikes in a source
    group are generated, certain neurons in the target group
    should have a value added to specific states. See
    Tutorial 2: Connections to understand this better.
    
    With arguments:
    
    ``source``
        The group from which spikes will be propagated.
    ``target``
        The group to which spikes will be propagated.
    ``state``
        The state variable name or number that spikes will be
        propagated to in the target group.
    ``delay``
        The delay between a spike being generated at the source
        and received at the target. Depending on the type of ``delay``
        it has different effects. If ``delay`` is a scalar value, then
        the connection will be initialised with all neurons having
        that delay. For very long delays, this may raise an error. If
        ``delay=True`` then the connection will be initialised as a
        :class:`DelayConnection`, allowing heterogeneous delays (a
        different delay for each synapse). ``delay`` can also be a
        pair ``(min,max)`` or a function of one or two variables, in
        both cases it will be initialised as a :class:`DelayConnection`,
        see the documentation for that class for details. Note that in
        these cases, initialisation of delays will only have the
        intended effect if used with the ``weight`` and ``sparseness``
        arguments below.
    ``max_delay``
        If you are using a connection with heterogeneous delays, specify
        this to set the maximum allowed delay (smaller values use less
        memory). The default is 5ms.
    ``modulation``
        The state variable name from the source group that scales
        the synaptic weights (for short-term synaptic plasticity).
    ``structure``
        Data structure: ``sparse`` (default), ``dense`` or
        ``dynamic``. See below for more information on structures.
    ``weight``
        If specified, the connection matrix will be initialised with
        values specified by ``weight``, which can be any of the values
        allowed in the methods `connect*`` below.
    ``sparseness``
        If ``weight`` is specified and ``sparseness`` is not, a full
        connection is assumed, otherwise random connectivity with this
        level of sparseness is assumed.
    
    **Methods**
    
    ``connect_random(P,Q,p[,weight=1[,fixed=False[,seed=None]]])``
        Connects each neuron in ``P`` to each neuron in ``Q`` with independent
        probability ``p`` and weight ``weight`` (this is the amount that
        gets added to the target state variable). If ``fixed`` is True, then
        the number of presynaptic neurons per neuron is constant. If ``seed``
        is given, it is used as the seed to the random number generators, for
        exactly repeatable results.
    ``connect_full(P,Q[,weight=1])``
        Connect every neuron in ``P`` to every neuron in ``Q`` with the given
        weight.
    ``connect_one_to_one(P,Q)``
        If ``P`` and ``Q`` have the same number of neurons then neuron ``i``
        in ``P`` will be connected to neuron ``i`` in ``Q`` with weight 1.
    ``connect(P,Q,W)``
        You can specify a matrix of weights directly (can be in any format
        recognised by NumPy). Note that due to internal implementation details,
        passing a full matrix rather than a sparse one may slow down your code
        (because zeros will be propagated as well as nonzero values).
        **WARNING:** No unit checking is done at the moment.

    Additionally, you can directly access the matrix of weights by writing::
    
        C = Connection(P,Q)
        print C[i,j]
        C[i,j] = ...
    
    Where here ``i`` is the source neuron and ``j`` is the target neuron.
    Note: if ``C[i,j]`` should be zero, it is more efficient not to write
    ``C[i,j]=0``, if you write this then when neuron ``i`` fires all the
    targets will have the value 0 added to them rather than just the
    nonzero ones.
    **WARNING:** No unit checking is currently done if you use this method.
    Take care to set the right units.
    
    **Connection matrix structures**
    
    Brian currently features three types of connection matrix structures,
    each of which is suited for different situations. Brian has two stages
    of connection matrix. The first is the construction stage, used for
    building a weight matrix. This stage is optimised for the construction
    of matrices, with lots of features, but would be slow for runtime
    behaviour. Consequently, the second stage is the connection stage,
    used when Brian is being run. The connection stage is optimised for
    run time behaviour, but many features which are useful for construction
    are absent (e.g. the ability to add or remove synapses). Conversion
    between construction and connection stages is done by the
    ``compress()`` method of :class:`Connection` which is called
    automatically when it is used for the first time.
    
    The structures are: 
    
    ``dense``
        A dense matrix. Allows runtime modification of all values. If
        connectivity is close to being dense this is probably the most
        efficient, but in most cases it is less efficient. In addition,
        a dense connection matrix will often do the wrong thing if
        using STDP. Because a synapse will be considered to exist but
        with weight 0, STDP will be able to create new synapses where
        there were previously none. Memory requirements are ``8NM``
        bytes where ``(N,M)`` are the dimensions. (A ``double`` float
        value uses 8 bytes.)
    ``sparse``
        A sparse matrix. See :class:`SparseConnectionMatrix` for
        details on implementation. This class features very fast row
        access, and slower column access if the ``column_access=True``
        keyword is specified (making it suitable for learning
        algorithms such as STDP which require this). Memory
        requirements are 12 bytes per nonzero entry for row access
        only, or 20 bytes per nonzero entry if column access is
        specified. Synapses cannot be created or deleted at runtime
        with this class (although weights can be set to zero).
    ``dynamic``
        A sparse matrix which allows runtime insertion and removal
        of synapses. See :class:`DynamicConnectionMatrix` for
        implementation details. This class features row and column
        access. The row access is slower than for ``sparse`` so this
        class should only be used when insertion and removal of
        synapses is crucial. Memory requirements are 24 bytes per
        nonzero entry. However, note that more memory than this
        may be required because memory is allocated using a
        dynamic array which grows by doubling its size when it runs
        out. If you know the maximum number of nonzero entries you will
        have in advance, specify the ``nnzmax`` keyword to set the
        initial size of the array. 
    
    **Advanced information**
    
    The following methods are also defined and used internally, if you are
    writing your own derived connection class you need to understand what
    these do.
    
    ``propagate(spikes)``
        Action to take when source neurons with indices in ``spikes``
        fired.
    ``do_propagate()``
        The method called by the :class:`Network` ``update()`` step,
        typically just propagates the spikes obtained by calling
        the ``get_spikes`` method of the ``source`` :class:`NeuronGroup`.
    '''
    #@check_units(delay=second)
    def __init__(self,source,target,state=0,delay=0*msecond,modulation=None,
                 structure='sparse',weight=None,sparseness=None,max_delay=5*ms,**kwds):
        if not isinstance(delay, float):
            if delay is True:
                delay = None # this instructs us to use DelayConnection, but not initialise any delays
            self.__class__ = DelayConnection
            self.__init__(source, target, state=state, modulation=modulation, structure=structure,
                          weight=weight, sparseness=sparseness, delay=delay, max_delay=max_delay, **kwds)
            return
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
        if not isinstance(self, DelayConnection):
            self.delay = int(delay/source.clock.dt) # Synaptic delay in time bins
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._keyword_based_init(weight=weight, sparseness=sparseness)
        
    def _keyword_based_init(self, weight=None, sparseness=None, **kwds):
        # Initialisation of weights
        # TODO: check consistency of weight and sparseness
        # TODO: select dense or sparse according to sparseness
        if weight is not None or sparseness is not None:
            if weight is None:
                weight = 1.0
            if sparseness is None:
                sparseness = 1
            if isinstance(weight, scipy.sparse.spmatrix) or isinstance(weight, ndarray):
                self.connect(W=weight)
            elif sparseness==1:
                self.connect_full(weight=weight)
            else:
                self.connect_random(weight=weight, p=sparseness)

    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
            # Target state variable
            sv=self.target._S[self.nstate]
            # If specified, modulation state variable
            if self._nstate_mod is not None:
                sv_pre = self.source._S[self._nstate_mod]
            # Get the rows of the connection matrix, each row will be either a
            # DenseConnectionVector or a SparseConnectionVector.
            rows = self.W.get_rows(spikes)
            if not self._useaccel: # Pure Python version is easier to understand, but slower than C++ version below
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        # Rows stored as sparse vectors without modulation
                        for row in rows:
                            sv[row.ind] += row
                    else:
                        # Rows stored as sparse vectors with modulation
                        for i, row in izip(spikes, rows):
                            # note we call the numpy __mul__ directly because row is
                            # a SparseConnectionVector with different mul semantics
                            sv[row.ind] += numpy.ndarray.__mul__(row, sv_pre[i])
                else:
                    if self._nstate_mod is None:
                        # Rows stored as dense vectors without modulation
                        for row in rows:
                            sv += row
                    else:
                        # Rows stored as dense vectors with modulation
                        for i, row in izip(spikes, rows):
                            sv += numpy.ndarray.__mul__(row, sv_pre[i])                    
            else: # C++ accelerated code, does the same as the code above but faster and less pretty
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowind = rowinds[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowind, "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyObject* _datasj = datas[j];
                                    PyArrayObject* _data = convert_to_numpy(_datasj, "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    int m = row.numElements();
                                    for(int k=0;k<m;k++)
                                    {
                                        sv(row(k)) += data(k);
                                    }
                                    Py_DECREF(_rowind);
                                    Py_DECREF(_datasj);
                                }
                                """
                        weave.inline(code,['sv','rowinds','datas','spikes','nspikes'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowind = rowinds[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowind, "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyObject* _datasj = datas[j];
                                    PyArrayObject* _data = convert_to_numpy(_datasj, "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    int m = row.numElements();
                                    double mod = sv_pre(spikes(j));
                                    for(int k=0;k<m;k++)
                                    {
                                        sv(row(k)) += data(k)*mod;
                                    }
                                    Py_DECREF(_rowind);
                                    Py_DECREF(_datasj);
                                }
                                """
                        weave.inline(code,['sv','sv_pre','rowinds','datas','spikes','nspikes'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                else:
                    if self._nstate_mod is None:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(sv)
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowsj = rows[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    for(int k=0;k<N;k++)
                                        sv(k) += row(k);
                                    Py_DECREF(_rowsj);
                                }
                                """
                        weave.inline(code,['sv','spikes','nspikes','N', 'rows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(sv)
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowsj = rows[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    double mod = sv_pre(spikes(j));
                                    for(int k=0;k<N;k++)
                                        sv(k) += row(k)*mod;
                                    Py_DECREF(_rowsj);
                                }
                                """
                        weave.inline(code,['sv','sv_pre','spikes','nspikes','N', 'rows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    
    def compress(self):
        if not self.iscompressed:
            self.W = self.W.connection_matrix()
            self.iscompressed = True

    def reinit(self):
        '''
        Resets the variables.
        '''
        pass

    def do_propagate(self):
        self.propagate(self.source.get_spikes(self.delay))
    
    def origin(self,P,Q):
        '''
        Returns the starting coordinate of the given groups in
        the connection matrix W.
        '''
        return (P._origin-self.source._origin,Q._origin-self.target._origin)

    # TODO: rewrite all the connection functions to work row by row for memory and time efficiency 

    # TODO: change this
    def connect(self,source=None,target=None,W=None):
        '''
        Connects (sub)groups P and Q with the weight matrix W (any type).
        Internally: inserts W as a submatrix.
        TODO: checks if the submatrix has already been specified.
        '''
        P=source or self.source
        Q=target or self.target
        i0,j0=self.origin(P,Q)
        self.W[i0:i0+len(P),j0:j0+len(Q)]=W
        
    def connect_random(self,source=None,target=None,p=1.,weight=1.,fixed=False, seed=None, sparseness=None):
        '''
        Connects the neurons in group P to neurons in group Q with probability p,
        with given weight (default 1).
        The weight can be a quantity or a function of i (in P) and j (in Q).
        If ``fixed`` is True, then the number of presynaptic neurons per neuron is constant.
        '''
        P=source or self.source
        Q=target or self.target
        if sparseness is not None: p = sparseness # synonym
        if seed is not None:
            numpy.random.seed(seed) # numpy's random number seed
            pyrandom.seed(seed) # Python's random number seed
        if fixed:
            random_matrix_function=random_matrix_fixed_column
        else:
            random_matrix_function=random_matrix
            
        if callable(weight):
            # Check units
            try:
                weight(0,0)+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
            self.connect(P,Q,random_matrix_function(len(P),len(Q),p,value=weight))
        else:
            # Check units
            try:
                weight+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
            self.connect(P,Q,random_matrix_function(len(P),len(Q),p,value=float(weight)))

    def connect_full(self,source=None,target=None,weight=1.):
        '''
        Connects the neurons in group P to all neurons in group Q,
        with given weight (default 1).
        The weight can be a quantity or a function of i (in P) and j (in Q).
        '''
        P=source or self.source
        Q=target or self.target
        # TODO: check units
        if callable(weight):
            # Check units
            try:
                weight(0,0)+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
            W=zeros((len(P),len(Q)))
            try:
                weight(0,1.*arange(0,len(Q)))
                failed=False
            except:
                failed= True
            if failed: # vector-based not possible
                log_debug('connections','Cannot build the connection matrix by rows')
                for i in range(len(P)):
                    for j in range(len(Q)):
                        w = float(weight(i,j))
                        #if not is_within_absolute_tolerance(w,0.,effective_zero):
                        W[i,j] = w
            else:
                for i in range(len(P)): # build W row by row
                    #w = weight(i,1.*arange(0,len(Q)))
                    #I = (abs(w)>effective_zero).nonzero()[0]
                    #print w, I, w[I]
                    #W[i,I] = w[I]
                    W[i,:] = weight(i,1.*arange(0,len(Q)))
            self.connect(P,Q,W)
        else:
            try:
                weight+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrect unit for the synaptic weights.",*inst._dims)
            self.connect(P,Q,float(weight)*ones((len(P),len(Q))))

    def connect_one_to_one(self,source=None,target=None,weight=1):
        '''
        Connects source[i] to target[i] with weights 1 (or weight).
        '''
        P=source or self.source
        Q=target or self.target
        if (len(P)!=len(Q)):
            raise AttributeError,'The connected (sub)groups must have the same size.'
        # TODO: unit checking
        self.connect(P,Q,float(weight)*eye_lil_matrix(len(P)))
        
    def __getitem__(self,i):
        return self.W.__getitem__(i)

    def __setitem__(self,i,x):
        # TODO: unit checking
        self.W.__setitem__(i,x)

class DelayConnection(Connection):
    '''
    Connection which implements heterogeneous postsynaptic delays
    
    Initialised as for a :class:`Connection`, but with the additional
    keyword:
    
    ``max_delay``
        Specifies the maximum delay time for any
        neuron. Note, the smaller you make this the less memory will be
        used.
    
    Overrides the following attribute of :class:`Connection`:
    
    .. attribute:: delay
    
        A matrix of delays. This array can be changed during a run,
        but at no point should it be greater than ``max_delay``.
    
    In addition, the methods ``connect``, ``connect_random``, ``connect_full``,
    and ``connect_one_to_one`` have a new keyword ``delay=...`` for setting the
    initial values of the delays, where ``delay`` can be one of:

    * A float, all delays will be set to this value
    * A pair (min, max), delays will be uniform between these two
      values.
    * A function of no arguments, will be called for each nonzero
      entry in the weight matrix.
    * A function of two argument ``(i,j)`` will be called for each
      nonzero entry in the weight matrix.
    * A matrix of an appropriate type (e.g. ndarray or lil_matrix).

    Finally, there is a method:
    
    ``set_delays(delay)``
        Where ``delay`` must be of one of the types above.
    
    **Notes**
    
    This class implements post-synaptic delays. This means that the spike is
    propagated immediately from the presynaptic neuron with the synaptic
    weight at the time of the spike, but arrives at the postsynaptic neuron
    with the given delay. At the moment, Brian only provides support for
    presynaptic delays if they are homogeneous, using the ``delay`` keyword
    of a standard ``Connection``.
    
    **Implementation**
    
    :class:`DelayConnection` stores an array of size ``(n,m)`` where
    ``n`` is ``max_delay/dt`` for ``dt`` of the target :class:`NeuronGroup`'s clock,
    and ``m`` is the number of neurons in the target. This array can potentially
    be quite large. Each row in this array represents the array that should be
    added to the target state variable at some particular future time. Which
    row corresponds to which time is tracked using a circular indexing scheme.
    
    When a spike from neuron ``i`` in the source is encountered, the delay time
    of neuron ``i`` is looked up, the row corresponding to the current time
    plus that delay time is found using the circular indexing scheme, and then
    the spike is propagated to that row as for a standard connection (although
    this won't be propagated to the target until a later time).
    
    **Warning**
    
    If you are using a dynamic connection matrix, it is your responsibility to
    ensure that the nonzero entries of the weight matrix and the delay matrix
    exactly coincide. This is not an issue for sparse or dense matrices.
    '''
       
    def __init__(self, source, target, state=0, modulation=None,
                 structure='sparse',
                 weight=None, sparseness=None, delay=None,
                 max_delay=5*msecond, **kwds):
        Connection.__init__(self, source, target, state=state, modulation=modulation,
                            structure=structure, weight=weight, sparseness=sparseness, **kwds)
        self._max_delay = int(max_delay/target.clock.dt)+1
        # Each row of the following array stores the cumulative effect of spikes at some
        # particular time, defined by a circular indexing scheme. The _cur_delay_ind attribute
        # stores the row corresponding to the current time, so that _cur_delay_ind+1 corresponds
        # to that time + target.clock.dt, and so on. When _cur_delay_ind reaches _max_delay it
        # resets to zero.
        self._delayedreaction = numpy.zeros((self._max_delay, len(target)))
        # vector of delay times, can be changed during a run
        if isinstance(structure,str):
            structure = construction_matrix_register[structure]
        self.delayvec = structure((len(source),len(target)),**kwds)
        self._cur_delay_ind = 0
        # this network operation is added to the Network object via the contained_objects
        # protocol (see the line after the function definition). The standard Connection.propagate
        # function propagates spikes to _delayedreaction rather than the target, and this
        # function which is called after the usual propagations propagates that data from
        # _delayedreaction to the target. It only needs to be called each target.clock update.
        @network_operation(clock=target.clock, when='after_connections')
        def delayed_propagate():
            # propagate from _delayedreaction -> target group
            target._S[self.nstate] += self._delayedreaction[self._cur_delay_ind, :]
            # reset the current row of _delayedreaction
            self._delayedreaction[self._cur_delay_ind, :] = 0.0
            # increase the index for the circular indexing scheme
            self._cur_delay_ind = (self._cur_delay_ind + 1) % self._max_delay
        self.contained_objects = [delayed_propagate]
        # this is just used to convert delayvec's which are in ms to integers, precalculating it makes it faster
        self._invtargetdt = 1/self.target.clock._dt
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        if delay is not None:
            self.set_delays(delay)
        
    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
            # Target state variable
            dr = self._delayedreaction
            # If specified, modulation state variable
            if self._nstate_mod is not None:
                sv_pre = self.source._S[self._nstate_mod]
            # Get the rows of the connection matrix, each row will be either a
            # DenseConnectionVector or a SparseConnectionVector.
            rows = self.W.get_rows(spikes)
            dvecrows = self.delayvec.get_rows(spikes)
            if not self._useaccel: # Pure Python version is easier to understand, but slower than C++ version below
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        # Rows stored as sparse vectors without modulation
                        for row, dvecrow in izip(rows, dvecrows):
                            if not len(row.ind)==len(dvecrow.ind):
                                raise RuntimeError('Weight and delay matrices must be kept in synchrony for sparse matrices.')
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            dr[drind, dvecrow.ind] += row
                    else:
                        # Rows stored as sparse vectors with modulation
                        for i, row, dvecrow in izip(spikes, rows, dvecrows):
                            if not len(row.ind)==len(dvecrow.ind):
                                raise RuntimeError('Weight and delay matrices must be kept in synchrony for sparse matrices.')
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            # note we call the numpy __mul__ directly because row is
                            # a SparseConnectionVector with different mul semantics
                            dr[drind, dvecrow.ind] += numpy.ndarray.__mul__(row, sv_pre[i])
                else:
                    if self._nstate_mod is None:
                        # Rows stored as dense vectors without modulation
                        drjind = numpy.arange(len(self.target), dtype=int)
                        for row, dvecrow in izip(rows, dvecrows):
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            dr[drind, drjind[:len(drind)]] += row
                    else:
                        # Rows stored as dense vectors with modulation
                        drjind = numpy.arange(len(self.target), dtype=int)
                        for i, row, dvecrow in izip(spikes, rows, dvecrows):
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            dr[drind, drjind[:len(drind)]] += numpy.ndarray.__mul__(row, sv_pre[i])                     
            else: # C++ accelerated code, does the same as the code above but faster and less pretty
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowind = rowinds[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowind, "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyObject* _datasj = datas[j];
                                    PyArrayObject* _data = convert_to_numpy(_datasj, "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    int m = row.numElements();
                                    for(int k=0;k<m;k++)
                                    {
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, row(k)) += data(k);
                                    }
                                    Py_DECREF(_rowind);
                                    Py_DECREF(_datasj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['rowinds','datas','spikes','nspikes',
                                           'dvecrows', 'dr', 'cdi', 'idt', 'md'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowind = rowinds[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowind, "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyObject* _datasj = datas[j];
                                    PyArrayObject* _data = convert_to_numpy(_datasj, "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    int m = row.numElements();
                                    double mod = sv_pre(spikes(j));
                                    for(int k=0;k<m;k++)
                                    {
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, row(k)) += data(k)*mod;
                                    }
                                    Py_DECREF(_rowind);
                                    Py_DECREF(_datasj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['sv_pre','rowinds','datas','spikes','nspikes',
                                           'dvecrows', 'dr', 'cdi', 'idt', 'md'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                else:
                    if self._nstate_mod is None:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(self.target)
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowsj = rows[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    for(int k=0;k<N;k++)
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, k) += row(k);
                                    Py_DECREF(_rowsj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['spikes','nspikes','N', 'rows',
                                           'dr','cdi','idt','md','dvecrows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(self.target)
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowsj = rows[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    double mod = sv_pre(spikes(j));
                                    for(int k=0;k<N;k++)
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, k) += row(k)*mod;
                                    Py_DECREF(_rowsj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['sv_pre','spikes','nspikes','N', 'rows',
                                           'dr','cdi','idt','md','dvecrows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])

    def do_propagate(self):
        self.propagate(self.source.get_spikes(0))
            
    def _set_delay_property(self, val):
        self.delayvec[:]=val

    delay = property(fget=lambda self:self.delayvec, fset=_set_delay_property)

    def compress(self):
        if not self.iscompressed:
            # We want delayvec to have nonzero entries at the same places as
            # W does, so we use W to initialise the compressed version of
            # delayvec, and then copy the values from the old delayvec to
            # the new compressed one, allowing delayvec and W to not have
            # to be perfectly intersected at the initialisation stage. If
            # the structure is dynamic, it will be the user's
            # responsibility to update them in sequence
            delayvec = self.delayvec
            self.delayvec = self.W.connection_matrix()
            for i in xrange(self.W.shape[0]):
                self.delayvec.set_row(i, array(todense(delayvec[i,:]), copy=False).flatten())    
            Connection.compress(self)
    
    def set_delays(self, delay):
        '''
        Set the delays corresponding to the weight matrix
        
        ``delay`` must be one of:
        
        * A float, all delays will be set to this value
        * A pair (min, max), delays will be uniform between these two
          values.
        * A function of no arguments, will be called for each nonzero
          entry in the weight matrix.
        * A function of two argument ``(i,j)`` will be called for each
          nonzero entry in the weight matrix.
        * A matrix of an appropriate type (e.g. ndarray or lil_matrix).
        '''
        if delay is None:
            return
        W = self.W
        if isinstance(W, sparse.lil_matrix):
            def getrow(i):
                return array(W.rows[i], dtype=int), W.data[i]
        else:
            def getrow(i):
                return slice(None), W[i,:]
        if isinstance(delay, float):
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                self.delayvec[i, inds] = delay
        elif isinstance(delay, (tuple, list)) and len(delays)==2:
            delaymin, delaymax = delay
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                rowdelay = rand(len(data))*(delaymax-delaymin)+delaymin
                self.delayvec[i, inds] = rowdelay
        elif callable(delay) and delay.func_code.co_argcount==0:
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                rowdelay = [delay() for _ in xrange(len(data))]
                self.delayvec[i, inds] = rowdelay
        elif callable(delay) and delay.func_code.co_argcount==2:
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                if isinstance(inds, slice) and inds==slice(None):
                    inds = numpy.arange(len(data))
                self.delayvec[i, inds] = delay(i, inds)
        else:
            #raise TypeError('delays must be float, pair or function of 0 or 2 arguments')
            self.delayvec[:,:] = delay # probably won't work, but then it will raise an error

    def connect(self, *args, **kwds):
        delay = kwds.pop('delay', None)
        Connection.connect(self, *args, **kwds)
        if delay is not None:
            self.set_delays(delay)

    def connect_random(self, *args, **kwds):
        delay = kwds.pop('delay', None)
        Connection.connect_random(self, *args, **kwds)
        if delay is not None:
            self.set_delays(delay)
    
    def connect_full(self, *args, **kwds):
        delay = kwds.pop('delay', None)
        Connection.connect_full(self, *args, **kwds)
        if delay is not None:
            self.set_delays(delay)

    def connect_one_to_one(self, *args, **kwds):
        delay = kwds.pop('delay', None)
        Connection.connect_one_to_one(self, *args, **kwds)
        if delay is not None:
            self.set_delays(delay)

class IdentityConnection(Connection):
    '''
    A :class:`Connection` between two groups of the same size, where neuron ``i`` in the
    source group is connected to neuron ``i`` in the target group.
    
    Initialised with arguments:
    
    ``source``, ``target``
        The source and target :class:`NeuronGroup` objects.
    ``state``
        The target state variable.
    ``weight``
        The weight of the synapse, must be a scalar.
    ``delay``
        Only homogeneous delays are allowed.
    
    The benefit of this class is that it has no storage requirements and is optimised for
    this special case.
    '''
    @check_units(delay=second)
    def __init__(self,source,target,state=0,weight=1,delay=0*msecond):
        if (len(source)!=len(target)):
            raise AttributeError,'The connected (sub)groups must have the same size.'
        self.source=source # pointer to source group
        self.target=target # pointer to target group
        if type(state)==types.StringType: # named state variable
            self.nstate=target.get_var_index(state)
        else:
            self.nstate=state # target state index
        self.W=float(weight) # weight
        source.set_max_delay(delay)
        self.delay=int(delay/source.clock.dt) # Synaptic delay in time bins
        #if self.delay>source._max_delay:
        #    raise AttributeError,"Transmission delay is too long."
        
    def propagate(self,spikes):
        '''
        Propagates the spikes to the target.
        '''
        self.target._S[self.nstate,spikes]+=self.W
        
    def compress(self):
        pass
    
class MultiConnection(Connection):
    '''
    A hub for multiple connections with a common source group.
    '''
    def __init__(self,source,connections=[]):
        self.source=source
        self.connections=connections
        self.iscompressed=False
        self.delay=connections[0].delay
        
    def propagate(self,spikes):
        '''
        Propagates the spikes to the targets.
        '''
        for C in self.connections:
            C.propagate(spikes)
            
    def compress(self):
        if not self.iscompressed:
            for C in self.connections:
                C.compress()
            self.iscompressed=True

# Generation of matrices
# TODO: vectorise
def random_matrix(n,m,p,value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    W=sparse.lil_matrix((n,m))
    if callable(value):
        if value.func_code.co_argcount==0: # TODO: should work with partial objects
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                W.rows[i]=sample(xrange(m),k)
                W.rows[i].sort()
                W.data[i]=[value() for _ in xrange(k)]
        elif value.func_code.co_argcount==2:
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                W.rows[i]=sample(xrange(m),k)
                W.rows[i].sort()
                W.data[i]=[value(i,j) for j in W.rows[i]]            
        else:
            raise AttributeError,"Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            k=random.binomial(m,p,1)[0]
            # Not significantly faster to generate all random numbers in one pass
            # N.B.: the sample method is implemented in Python and it is not in Scipy
            W.rows[i]=sample(xrange(m),k)
            W.rows[i].sort()
            W.data[i]=[value]*k

    return W

def random_matrix_fixed_column(n,m,p,value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    The number of non-zero entries by per column is fixed: (int)(p*n)
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    W=sparse.lil_matrix((n,m))
    k=(int)(p*n)
    for j in xrange(m):
        # N.B.: the sample method is implemented in Python and it is not in Scipy
        for i in sample(xrange(n),k):
            W.rows[i].append(j)
            
    if callable(value):
        if value.func_code.co_argcount==0:
            for i in xrange(n):
                W.data[i]=[value() for _ in xrange(len(W.rows[i]))]
        elif value.func_code.co_argcount==2:
            for i in xrange(n):
                W.data[i]=[value(i,j) for j in W.rows[i]]            
        else:
            raise AttributeError,"Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            W.data[i]=[value]*len(W.rows[i])

    return W

# Generation of matrices row by row
# TODO: vectorise
def random_matrix_row_by_row(n,m,p,value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    if callable(value):
        if value.func_code.co_argcount==0:
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                row = sample(xrange(m),k)
                row.sort()
                yield row, [value() for _ in xrange(k)]
        elif value.func_code.co_argcount==2:
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                row = sample(xrange(m),k)
                row.sort()
                yield row, [value(i,j) for j in W.rows[i]]
        else:
            raise AttributeError,"Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            k=random.binomial(m,p,1)[0]
            row = sample(xrange(m),k)
            row.sort()
            yield row, value 


def eye_lil_matrix(n):
    '''
    Returns the identity matrix of size n as a lil_matrix
    (sparse matrix).
    '''
    M=sparse.lil_matrix((n,n))
    M.setdiag([1.]*n)
    return M

def _define_and_test_interface(self):
    '''
    :class:`Connection`
    ~~~~~~~~~~~~~~~~~~~
    
    **Initialised as:** ::
    
        Connection(source, target[, state=0[, delay=0*ms]])
    
    With arguments:
    
    ``source``
        The group from which spikes will be propagated.
    ``target``
        The group to which spikes will be propagated.
    ``state``
        The state variable name or number that spikes will be
        propagated to in the target group.
    ``delay``
        The delay between a spike being generated at the source
        and received at the target. At the moment, the mechanism
        for delays only works for relatively short delays (an
        error will be generated for delays that are too long), but
        this is subject to change. The exact behaviour then is
        not part of the assured interface, although it is very
        likely that the syntax will not change (or will at least
        be backwards compatible).
    
    **Methods**
    
    ``connect_random(P,Q,p[,weight=1])``
        Connects each neuron in ``P`` to each neuron in ``Q``.
    ``connect_full(P,Q[,weight=1])``
        Connect every neuron in ``P`` to every neuron in ``Q``.
    ``connect_one_to_one(P,Q)``
        If ``P`` and ``Q`` have the same number of neurons then neuron ``i``
        in ``P`` will be connected to neuron ``i`` in ``Q`` with weight 1.
    
    Additionally, you can directly access the matrix of weights by writing::
    
        C = Connection(P,Q)
        print C[i,j]
        C[i,j] = ...
    
    Where here ``i`` is the source neuron and ``j`` is the target neuron.
    Note: No unit checking is currently done if you use this method,
    but this is subject to change for future releases.

    The behaviour when a list of neuron ``spikes`` is received is to
    add ``W[i,:]`` to the target state variable for each ``i`` in ``spikes``. 
    '''
    
    from directcontrol import SpikeGeneratorGroup
    from neurongroup import NeuronGroup
    from network import Network
    from utils.approximatecomparisons import is_approx_equal
    from clock import reinit_default_clock
    
    # test Connection object
    
    eqs = '''
    da/dt = 0.*hertz : 1.
    db/dt = 0.*hertz : 1.
    '''
    
    spikes = [(0,1*msecond),(1,3*msecond)]
    
    G1 = SpikeGeneratorGroup(2,spikes)
    G2 = NeuronGroup(2,model=eqs,threshold=10.,reset=0.)
    
    # first test the methods
    # connect_full
    C = Connection(G1,G2)
    C.connect_full(G1, G2, weight=2.)
    for i in range(2):
        for j in range(2):
            self.assert_(is_approx_equal(C[i,j],2.))
    # connect_random
    C = Connection(G1,G2)
    C.connect_random(G1, G2, 0.5, weight=2.)
    # can't assert anything about that
    # connect_one_to_one
    C = Connection(G1,G2)
    C.connect_one_to_one(G1, G2)
    for i in range(2):
        for j in range(2):
            if i==j:
                self.assert_(is_approx_equal(C[i,j],1.))
            else:
                self.assert_(is_approx_equal(C[i,j],0.))
    del C
    # and we will use a specific set of connections in the next part
    Ca = Connection(G1,G2,'a')
    Cb = Connection(G1,G2,'b')
    Ca[0,0]=1.
    Ca[0,1]=1.
    Ca[1,0]=1.
    #Ca[1,1]=0 by default
    #Cb[0,0]=0 by default
    Cb[0,1]=1.
    Cb[1,0]=1.
    Cb[1,1]=1.
    net = Network(G1,G2,Ca,Cb)
    net.run(2*msecond)
    # after 2 ms, neuron 0 will have fired, so a 0 and 1 should
    # have increased by 1 to [1,1], and b 1 should have increased
    # by 1 to 1
    self.assert_(is_approx_equal(G2.a[0],1.))
    self.assert_(is_approx_equal(G2.a[1],1.))
    self.assert_(is_approx_equal(G2.b[0],0.))
    self.assert_(is_approx_equal(G2.b[1],1.))
    net.run(2*msecond)
    # after 4 ms, neuron 1 will have fired, so a 0 should have
    # increased by 1 to 2, and b 0 and 1 should have increased
    # by 1 to [1, 2]
    self.assert_(is_approx_equal(G2.a[0],2.))
    self.assert_(is_approx_equal(G2.a[1],1.))
    self.assert_(is_approx_equal(G2.b[0],1.))
    self.assert_(is_approx_equal(G2.b[1],2.))
    
    reinit_default_clock()

# this is put down here because it is needed in the code above but the order of imports is messed up
# if it is included at the top
from network import network_operation