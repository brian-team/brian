from base import *

__all__ = [
         'ConnectionVector', 'SparseConnectionVector', 'DenseConnectionVector',
         ]

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
        # Note that removing this check is potentially dangerous, but only in weird circumstances would it cause
        # any problems, and leaving it in causes problems for STDP with DelayConnection (because the indices are
        # not the same, but should be presumed to be equal).
        #if other.ind is not self.ind:
        #    raise TypeError('__add__(SparseConnectionVector, SparseConnectionVector) only defined if indices are the same')
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
