from brian import *
import numpy

class PureQuantityBase(Quantity):
    def __init__(self, value):
        self.dim = value.dim
    def __div__(self, other):
        try:
            return Quantity.__div__(self, other)
        except ZeroDivisionError:
            return Quantity.with_dimensions(0, self.dim/other.dim)
    __truediv__ = __div__
    def __rdiv__(self, other):
        try:
            return Quantity.__div__(other, self)
        except ZeroDivisionError:
            return Quantity.with_dimensions(0, other.dim/self.dim)
    __rtruediv__ = __rdiv__
    def __mod__(self, other):
        try:
            return Quantity.__mod__(self, other)
        except ZeroDivisionError:
            return Quantity.with_dimensions(0, self.dim)

def returnpure(meth):
    def f(*args, **kwds):
        x = meth(*args, **kwds)
        if isinstance(x, Quantity) and not isinstance(x, PureQuantity):
            return PureQuantity(x)
        else:
            return x
    return f

class PureQuantity(PureQuantityBase):
    for methname in dir(PureQuantityBase):
        meth = getattr(PureQuantityBase, methname)
        try:
            meth2 = getattr(numpy.float64, methname)
        except AttributeError:
            meth2 = meth
        if callable(meth) and meth is not meth2:
            exec methname+'=returnpure(PureQuantityBase.'+methname+')'
    del meth, meth2, methname

def namespace_replace_quantity_with_pure(ns):
    newns = {}
    for k, v in ns.iteritems():
        if isinstance(v, Quantity):
            v = PureQuantity(v)
        newns[k] = v
    return newns

if __name__=='__main__':
    ns = {'volt':volt, 'amp':1*amp}
    ns = namespace_replace_quantity_with_pure(ns)
    print ns['volt'].__class__
    exec 'print 1*volt/(1*amp-1*amp)' in ns
    