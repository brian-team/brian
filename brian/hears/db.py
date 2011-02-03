from brian import *
from numpy import *

__all__ = ['dB', 'dB_type', 'dB_error', 'gain']

class dB_error(ValueError):
    '''
    Error raised when values in dB are used inconsistently with other units.
    '''
    pass

class dB_type(float64):
    '''
    The type of values in dB.
    
    dB values are assumed to be RMS dB SPL assuming that the sound source is
    measured in Pascals.
    '''
    def __str__(self):
        return str(float(self))+'*dB'
    def __repr__(self):
        return repr(float(self))+'*dB'
    def __mul__(self, other):
        if isinstance(other, dB_type):
            raise dB_error('Cannot multiply dB by dB')
        return dB_type(float(self)*other)
    __rmul__ = __mul__
    def __div__(self, other):
        if isinstance(other, dB_type):
            raise dB_error('Cannot divide dB by dB')
        return dB_type(float(self)/other)
    __truediv__ = __div__
    def __rdiv__(self, other):
        if isinstance(other, dB_type):
            raise dB_error('Cannot divide dB by dB')
        return dB_type(other/float(self))
    __rtruediv__ = __rdiv__
    def __add__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Cannot add a dB object to a non-dB object')
        return dB_type(float(self)+float(other))
    __radd__ = __add__
    def __sub__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Cannot subtract a dB object from a non-dB object')
        return dB_type(float(self)-float(other))
    def __rsub__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Cannot subtract a dB object from a non-dB object')
        return dB_type(float(other)-float(self))
    def __neg__(self):
        return dB_type(-float(self))
    def __pos__(self):
        return self
    def __abs__(self):
        return dB_type(abs(float(self)))
    def __pow__(self, other):
        raise dB_error('Cannot take powers with dB')
    __rpow__ = __pow__
    def __lt__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Can only compare with another dB')
        return float(self)<float(other)
    def __le__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Can only compare with another dB')
        return float(self)<=float(other)
    def __gt__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Can only compare with another dB')
        return float(self)>float(other)
    def __ge__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Can only compare with another dB')
        return float(self)>=float(other)
    def __eq__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Can only compare with another dB')
        return float(self)==float(other)
    def __ne__(self, other):
        if not isinstance(other, dB_type):
            raise dB_error('Can only compare with another dB')
        return float(self)!=float(other)
    def __reduce__(self):
        return (dB_type, (float(self),))
    def gain(self):
        return 10**(float(self)/20.0)


dB = dB_type(1.0)

def gain(level):
    '''
    Returns the gain factor associated to a level in dB.
    
    The formula is:
    
    gain = 10**(level/20.0)
    '''
    if not isinstance(level, dB_type):
        raise dB_error('Level must be in dB')
    return level.gain()
