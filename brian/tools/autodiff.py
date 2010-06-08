'''
Automatic differentiation.
(C) R. Brette 2008
This file is released under the terms of the BSD licence
--------------------------------------------------------
Method: forward accumulation by operator overloading.

Use::
  df=differentiate(f)                  # where f is a single-variable function
  y=differentiate(f,x0)                # differentiate at x=x0
  y=differentiate(f,x0,order=2)        # second-order derivative at x=x0
  y=differentiate(f,(a,b),order=(1,2)) # d2f/dxdy(a,b)
  y=differentiate(f,order=(1,2))       # d2f/dxdy
  
  a0,a1,a2=taylor_series(f,x0,order=2) # 2nd order Taylor series of f at x0
  J=gradient(f,(a,b))                  # gradient of f at (a,b) (returns a list)
  H=hessian(f,(a,b))                   # hessian of f at (a,b) (returns a numpy array)
  
TODO:
* replace x by x0
* merge HigherDifferentiable in Differentiable?
* simplify code
* gradient, taylor_series, hessian: x0 should be optional

* iadd, etc. (more operators)
* NonDifferentiableException and maybe use this on conditions (when cmp=0)
* Differential operators: Laplacian...
* gradient, taylor_series, hessian: x0 should be optional
* test with units, functions Rp->something, alternative differential operators,
  differential (f:Rp->Rn), complex numbers, arithmetic derivative
* remove most dependency with numpy (except Hessian)
* implicitly defined functions

If several partial derivatives of the same function need to be
calculated, it is faster to use the lower level object Differentiable.
Example::
  y=f(Differentiable('x',3),Differentiable('y',2))
will return a Differentiable object with y.val=f(3,2)
and y.diff={'x': df/dx(3,2), 'y': df/dy(3,2)}
'''

from numpy import exp, sin, cos, log, zeros # todo: look in frame instead?
import types

__all__ = ['differentiate', 'Differentiable', 'taylor_series', 'gradient', 'hessian']

def differentiate(f, x=None, order=1):
    '''
    Calculate the derivative of f at point x.
    Higher-order derivatives are possible.
    Ex.: differentiate(lambda x:3*x*x+2,x=2,order=2)
    (Returns: 6.0)
    
    Partial derivatives:
    1) Give a tuple with the order.
    Ex.: differentiate(lambda x,y:x*y+2*y+1,x=(1,2),order=(0,1))
    (Returns: 3.0 (partial derivative d/dy) )
    2) Give a dictionnary with the order:
    Ex.: differentiate(lambda x,y:x*y+2*y+1,x=(1,2),order={'y':1})
    '''
    if x is None:
        return lambda x:differentiate(f, x, order)
    if (type(order) == types.ListType) or (type(order) == types.TupleType): # several variables
        # Build the list of arguments
        n = sum(order) # total order
        args = [HigherDifferentiable(i, x[i], n) for i in range(len(x))]
        y = f(*args)
        for i in xrange(len(x)):
            for _ in xrange(order[i]):
                if isinstance(y, Differentiable) and (i in y.diff):
                    y = y.diff[i]
                else: # constant
                    return 0.
        return y
    elif (type(order) == types.DictType): # named variables
        # Build the list of arguments
        n = sum(order.itervalues())
        vars = list(f.func_code.co_varnames)
        args = []
        args.extend(x)
        for name in order.iterkeys():
            i = vars.index(name)
            args[i] = HigherDifferentiable(name, x[i], n)
        y = f(*args)
        for name, n in order.iteritems():
            for j in range(n):
                if isinstance(y, Differentiable) and (name in y.diff):
                    y = y.diff[name]
                else: # constant
                    return 0.
        return y
    elif order == 0:
        return f(x)
    elif order == 1:
        y = f(Differentiable('x', x))
        if isinstance(y, Differentiable) and ('x' in y.diff):
            return y.diff['x']
        else: # constant
            return 0.
    else:
        return differentiate(lambda y:differentiate(f, y, order=order - 1), x)

def taylor_series(f, x, order=1):
    '''
    Returns the list of coefficients of the Taylor Series of f
    at x (scalar).
    '''
    y = f(HigherDifferentiable('x', x, order))
    series = []
    n = 1.
    for i in range(order + 1):
        # Value
        val = y
        while isinstance(val, Differentiable):
            val = val.val
        series.append(val / n)
        n *= (i + 1)
        # Next order
        if isinstance(y, Differentiable) and ('x' in y.diff):
            y = y.diff['x']
        else:
            y = 0.
    return series

def gradient(f, x):
    '''
    Gradient of f at x (= tuple).
    Result = list.
    '''
    args = [Differentiable(i, val=x[i]) for i in range(len(x))]
    y = f(*args)
    if isinstance(y, Differentiable):
        result = []
        for i in range(len(x)):
            if i in y.diff:
                result.append(y.diff[i])
            else:
                result.append(0.)
        return result
    else: # constant
        return [0.] * len(x)

def hessian(f, x):
    '''
    Hessian of f at x (= tuple).
    Result = array.
    N.B.: not working with vectors.
    '''
    args = [HigherDifferentiable(i, x[i], 2) for i in range(len(x))]
    y = f(*args)
    result = zeros((len(x), len(x)))
    if isinstance(y, Differentiable): # non-zero Hessian
        for i in range(len(x)):
            if i in y.diff and isinstance(y.diff[i], Differentiable):
                for j in range(len(x)):
                    if j in y.diff[i].diff:
                        result[i, j] = y.diff[i].diff[j]
    return result


class Differentiable(object):
    '''
    A differentiable variable.
    Implemented operations: +,-,*,[],len,exp,sin,cos,/,abs,sqrt,comparisons,**
    '''
    def __init__(self, name=None, val=0.):
        '''
        Initializes a variable and its derivative.
        If the name is None, then it is a constant.
        '''
        self.val = val # value
        if name == None: # constant
            self.diff = {} # derivative
        else:
            self.diff = {name:1.} # or jacobian? (eye)

    def sqrt(self):
        return self ** .5

    def __cmp__(self, x):
        if isinstance(x, Differentiable):
            return cmp(self.val, x.val)
        else:
            return cmp(self.val, x)

    def __abs__(self):
        # NOT WORKING WITH VECTORS
        if self.val == 0:
            # TODO: specific exception
            raise ZeroDivisionError, "x:abs(x) is not differentiable at x=0."
        elif self.val > 0:
            return self
        else:
            return - self

    def __add__(self, y):
        # VECTOR-READY
        if isinstance(y, Differentiable):
            zdict = {}
            zdict.update(self.diff)
            zdict.update(y.diff)
            for key in zdict.iterkeys():
                if (key in self.diff) and (key in y.diff):
                    zdict[key] = self.diff[key] + y.diff[key]
            z = Differentiable()
            z.val = self.val + y.val
            z.diff = zdict
            return z
        else: # Adding a constant
            z = Differentiable(val=self.val + y)
            z.diff = self.diff
            return z

    def __radd__(self, x):
        # VECTOR-READY
        if isinstance(x, Differentiable):
            zdict = {}
            zdict.update(self.diff)
            zdict.update(x.diff)
            for key in zdict.iterkeys():
                if (key in self.diff) and (key in x.diff):
                    zdict[key] = x.diff[key] + self.diff[key]
            z = Differentiable()
            z.val = x.val + self.val
            z.diff = zdict
            return z
        else: # Adding a constant
            z = Differentiable(val=x + self.val)
            z.diff = self.diff
            return z

    def __mul__(self, y):
        # !Not working with vectors!
        if isinstance(y, Differentiable):
            #TODO: with vectors
            zdict = {}
            zdict.update(self.diff)
            zdict.update(y.diff)
            for key in zdict.iterkeys():
                if (key in self.diff) and (key in y.diff):
                    zdict[key] = self.diff[key] * y.val + self.val * y.diff[key]
                elif (key in self.diff):
                    zdict[key] = self.diff[key] * y.val
                else:
                    zdict[key] = self.val * y.diff[key]
            z = Differentiable()
            z.val = self.val * y.val
            z.diff = zdict
            return z
        else: # Multiplying by a constant
            z = Differentiable(val=self.val * y)
            for key in self.diff:
                z.diff[key] = self.diff[key] * y
            return z

    def __rmul__(self, x):
        if isinstance(x, Differentiable):
            #TODO: with vectors
            zdict = {}
            zdict.update(self.diff)
            zdict.update(x.diff)
            for key in zdict.iterkeys():
                if (key in self.diff) and (key in x.diff):
                    zdict[key] = x.val * self.diff[key] + x.diff[key] * self.val
                elif (key in self.diff):
                    zdict[key] = x.val * self.diff[key]
                else:
                    zdict[key] = x.diff[key] * self.val
            z = Differentiable()
            z.val = x.val * self.val
            z.diff = zdict
            return z
        else: # Multiplying by a constant
            z = Differentiable(val=x * self.val)
            for key in self.diff:
                z.diff[key] = x * self.diff[key]
            return z

    def __div__(self, x):
        return self * (x ** -1)

    def __rdiv__(self, x):
        return x * (self ** -1)

    def __sub__(self, y):
        # VECTOR-READY
        if isinstance(y, Differentiable):
            zdict = {}
            zdict.update(self.diff)
            zdict.update(y.diff)
            for key in zdict.iterkeys():
                if (key in self.diff) and (key in y.diff):
                    zdict[key] = self.diff[key] - y.diff[key]
            z = Differentiable()
            z.val = self.val - y.val
            z.diff = zdict
            return z
        else: # Subtracting a constant
            z = Differentiable(val=self.val - y)
            z.diff = self.diff
            return z

    def __rsub__(self, x):
        # VECTOR-READY
        if isinstance(x, Differentiable):
            zdict = {}
            zdict.update(self.diff)
            zdict.update(x.diff)
            for key in zdict.iterkeys():
                if (key in self.diff) and (key in x.diff):
                    zdict[key] = x.diff[key] - self.diff[key]
            z = Differentiable()
            z.val = x.val - self.val
            z.diff = zdict
            return z
        else: # Subtracting a constant
            z = Differentiable(val=x - self.val)
            for key in self.diff:
                z.diff[key] = -self.diff[key]
            return z

    def __neg__(self):
        # VECTOR-READY
        z = Differentiable(val= -self.val)
        for key in self.diff:
            z.diff[key] = -self.diff[key]
        return z

    def __pow__(self, x):
        # NOT WORKING WITH VECTORS
        if isinstance(x, Differentiable):
            return exp(x * log(self))
        elif x == 0:
            z = Differentiable(val=self.val ** x)
            for key in self.diff:
                z.diff[key] = 0 * self.diff[key]
            return z
        else:
            z = Differentiable(val=self.val ** x)
            for key in self.diff:
                z.diff[key] = x * self.val ** (x - 1) * self.diff[key]
            return z

    def __rpow__(self, x):
        # NOT WORKING WITH VECTORS
        return exp(self * log(x))

    def __getitem__(self, i):
        z = Differentiable(val=self.val[i])
        z.diff = {}
        for key in self.diff.iterkeys():
            z.diff[key] = self.diff[key][i, :]
        return z

    def __len__(self):
        return len(self.val)

    def __str__(self):
        s = str(self.val)
        for key, value in self.diff.iteritems():
            s += ' ; d/d' + key + '=' + str(value)
        return s

    def exp(self):
        # !! Not working for vectors !!
        zdict = {}
        zdict.update(self.diff)
        for key in zdict.iterkeys():
            zdict[key] = exp(self.val) * self.diff[key]
        z = Differentiable(val=exp(self.val))
        z.diff = zdict
        return z

    def cos(self):
        # !! Not working for vectors !!
        zdict = {}
        zdict.update(self.diff)
        for key in zdict.iterkeys():
            zdict[key] = -sin(self.val) * self.diff[key]
        z = Differentiable(val=cos(self.val))
        z.diff = zdict
        return z

    def sin(self):
        # !! Not working for vectors !!
        zdict = {}
        zdict.update(self.diff)
        for key in zdict.iterkeys():
            zdict[key] = cos(self.val) * self.diff[key]
        z = Differentiable(val=sin(self.val))
        z.diff = zdict
        return z

    def log(self):
        # !! Not working for vectors !!
        zdict = {}
        zdict.update(self.diff)
        for key in zdict.iterkeys():
            zdict[key] = (1. / (self.val)) * self.diff[key]
        z = Differentiable(val=log(self.val))
        z.diff = zdict
        return z


class HigherDifferentiable(Differentiable):
    '''
    A differentiable variable with order n.
    '''
    def __init__(self, name=None, val=0., order=1):
        Differentiable.__init__(self, name, val)
        if order > 1:
            self.val = HigherDifferentiable(name, val, order - 1)

if __name__ == '__main__':
    print "Derivative:"
    print "> differentiate(lambda x:3*x*x+2,x=2,order=2)"
    print differentiate(lambda x:3 * x * x + 2, x=2, order=2)
    print "> differentiate(lambda x:3*x*x+2,order=2)(2)"
    print differentiate(lambda x:3 * x * x + 2, order=2)(2)
    print "Partial derivative:"
    print "> differentiate(lambda x,y:x*y+2*y+1,x=(1,2),order=(0,1))"
    print differentiate(lambda x, y:x * y + 2 * y + 1, x=(1, 2), order=(0, 1))
