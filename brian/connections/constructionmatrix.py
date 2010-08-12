from base import *
from sparsematrix import *
from connectionmatrix import *

__all__ = [
         'ConstructionMatrix',
         'SparseConstructionMatrix',
         'DenseConstructionMatrix',
         'DynamicConstructionMatrix',
         'construction_matrix_register',
         ]

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
        self[:] = 0
        self.init_kwds = kwds

    def connection_matrix(self, **additional_kwds):
        self.init_kwds.update(additional_kwds)
        kwds = self.init_kwds
        return DenseConnectionMatrix(self, **kwds)

    def __setitem__(self, index, W):
        # Make it work for sparse matrices
        #if isinstance(W,sparse.spmatrix):
        if isinstance(W, sparse.spmatrix):
            ndarray.__setitem__(self, index, W.todense())
        else:
            ndarray.__setitem__(self, index, W)

    def todense(self):
        return asarray(self)

class SparseConstructionMatrix(ConstructionMatrix, SparseMatrix):
    '''
    SparseConstructionMatrix is converted to SparseConnectionMatrix.
    '''
    def __init__(self, arg, **kwds):
        SparseMatrix.__init__(self, arg)
        self.init_kwds = kwds

    def connection_matrix(self, **additional_kwds):
        self.init_kwds.update(additional_kwds)
        return SparseConnectionMatrix(self, **self.init_kwds)


class DynamicConstructionMatrix(ConstructionMatrix, SparseMatrix):
    '''
    DynamicConstructionMatrix is converted to DynamicConnectionMatrix.
    '''
    def __init__(self, arg, **kwds):
        SparseMatrix.__init__(self, arg)
        self.init_kwds = kwds

    def connection_matrix(self, **additional_kwds):
        self.init_kwds.update(additional_kwds)
        return DynamicConnectionMatrix(self, **self.init_kwds)

# this is used to look up str->class conversions for structure=... keyword
construction_matrix_register = {
        'dense':DenseConstructionMatrix,
        'sparse':SparseConstructionMatrix,
        'dynamic':DynamicConstructionMatrix,
        }
