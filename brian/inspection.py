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
'''
Inspection of strings with python statements 
and models defined by differential equations.

TODO: some of the module is obsolete
'''

__all__ = ['is_affine', 'depends_on', 'Term', 'get_global_term',
         'get_var_names', 'check_equations_units', 'fill_vars', 'AffineFunction',
         'get_identifiers', 'modified_variables', 'namespace', 'clean_text',
         'namespace_replace_quantity_with_pure', 'list_replace_quantity_with_pure']

import numpy
from numpy import array
from units import *
import parser
import re
import inspect
from copy import copy


class PureQuantityBase(Quantity):
    def __init__(self, value):
        self.dim = value.dim

    def __div__(self, other):
        if not isinstance(other, Quantity) and not is_scalar_type(other):
            return NotImplemented
        try:
            return Quantity.__div__(self, other)
        except ZeroDivisionError:
            try:
                odim = other.dim
            except AttributeError:
                odim = Dimension()
            return Quantity.with_dimensions(0, self.dim / odim)
    __truediv__ = __div__
    def __rdiv__(self, other):
        if not isinstance(other, Quantity) and not is_scalar_type(other):
            return NotImplemented
        try:
            return Quantity.__rdiv__(self, other)
        except ZeroDivisionError:
            try:
                odim = other.dim
            except AttributeError:
                odim = Dimension()
            return Quantity.with_dimensions(0, odim / self.dim)
    __rtruediv__ = __rdiv__
    def __mod__(self, other):
        if not isinstance(other, Quantity) and not is_scalar_type(other):
            return NotImplemented
        try:
            return Quantity.__mod__(self, other)
        except ZeroDivisionError:
            return Quantity.with_dimensions(0, self.dim)

def returnpure(meth):
    def f(*args, **kwds):
        x = meth(*args, **kwds)
        if isinstance(x, Quantity):
            return PureQuantity(x)
        else:
            return x
    return f


class PureQuantity(PureQuantityBase):
    '''
    Use this class for unit checking.
    
    The idea is that operations should always work if they are
    dimensionally consistent regardless of the values. The key one is that
    division by zero does not raise an error but returns a value with the
    correct dimensions (in fact, it returns 0 with the correct dimensions).
    This is important for Brian because we do unit checking by substituting
    zeros into the equations, which sometimes gives a divide by zero error.
    
    The way it works is that it derives from Quantity, but for division it
    wraps a try: except ZeroDivisionError: around the operation, and returns
    a zero with the correct units if it encounters it. In addition to that,
    it wraps every method of Quantity so that they return PureQuantity objects
    instead of Quantity objects (otherwise e.g. a*b would be a Quantity not
    a PureQuantity even if a, b were PureQuantity).
    
    Finally, the *_replace_quantity_with_pure functions are just designed to
    scan through a dict or list of variables and replace Quantity objects
    by PureQuantity objects. You need to do this when evaluating some code
    in a user namespace, for example.
    '''
    for methname in dir(PureQuantityBase):
        meth = getattr(PureQuantityBase, methname)
        try:
            meth2 = getattr(numpy.float64, methname)
        except AttributeError:
            meth2 = meth
        if callable(meth) and meth is not meth2:
            exec methname + '=returnpure(PureQuantityBase.' + methname + ')'
    del meth, meth2, methname

def namespace_replace_quantity_with_pure(ns):
    newns = {}
    for k, v in ns.iteritems():
        if isinstance(v, Quantity):
            v = PureQuantity(v)
        newns[k] = v
    return newns

def list_replace_quantity_with_pure(L):
    newL = []
    for v in L:
        if isinstance(v, Quantity):
            v = PureQuantity(v)
        newL.append(v)
    return newL

def namespace(expr, level=0, return_unknowns=False):
    '''
    Returns a namespace with the values of identifiers in expr,
    taking from:
    * local namespace
    * global namespace
    * units
    '''
    # Build the namespace
    frame = inspect.stack()[level + 1][0]
    global_namespace, local_namespace = frame.f_globals, frame.f_locals
    # Find external objects
    space = {}
    unknowns = []
    for var in get_identifiers(expr):
        if var in local_namespace: #local
            space[var] = local_namespace[var]
        elif var in global_namespace: #global
            space[var] = global_namespace[var]
        elif var in globals(): # typically units
            space[var] = globals()[var]
        else:
            unknowns.append(var)
    if return_unknowns:
        return space, unknowns
    else:
        return space

def get_identifiers(expr):
    '''
    Returns the list of identifiers (variables or functions) in the
    Python expression (string).
    '''
    # cleaner: parser.expr(expr).tolist() then find leaves of the form [1,name]
    return parser.suite(expr).compile().co_names

def clean_text(expr):
    '''
    Cleans a Python expression or statement:
    * Remove comments (# comment)
    * Merge multi-line statements (\)
    * Split at semi-columns (careful: indentation is ignored)
    '''
    # Merge multi-line statements
    expr = re.sub('\\\s*?\n', ' ', expr)
    # Remove comments
    expr = re.sub('#.*', '', expr)
    # Split at semi-columns
    expr = re.sub(';', '\n', expr)
    return expr

def modified_variables(expr):
    '''
    Returns the list of variables or functions in expr that are in left-hand sides, e.g.:
        x+=5
    expr can be a multiline statement.
    Multiline comments are not allowed but multiline statements are.
    Functions may also be returned as in:
        do(something) # here do is returned, not something
        
    TODO: maybe functions should be removed?
    TODO: better handling of semi-columns (;)
    '''
    vars = get_identifiers(expr)
    expr = clean_text(expr)
    # Find lines that start by an identifier
    mod_vars = []
    for line in expr.splitlines():
        s = re.search(r'^\s*(\w+)\b', line)
        if s and (s.group(1) in vars):
            mod_vars.append(s.group(1))
    return mod_vars

def fill_vars(f, keepnamespace=False, *varnames):
    '''
    Returns a function with arguments given by varnames (list or tuple),
    given that the arguments of f are in varnames.
    If keepnamespace is True, then the original func_globals dictionary of
    the function is kept.
    Purpose: changing the syntax for function calls.
    Example:
    f=lambda x:2*x
    g=fill_vars(f,'y','x')
    Then g(1,2) returns f(2).
    This is somehow the inverse of partial (in module functools).
    N.B.: the order of variables matters.
    '''
    if list(f.func_code.co_varnames) == varnames:
        return f
    varstring = varnames[0]
    for name in varnames[1:]:
        varstring += ',' + name
    shortvarstring = f.func_code.co_varnames[0]
    for name in f.func_code.co_varnames[1:]:
        shortvarstring += ',' + name
    if keepnamespace:
        # Create a unique name
        fname = 'fill_vars_function' + id(f)
        f.func_globals[fname] = f
        return eval('lambda ' + varstring + ': ' + fname + '(' + shortvarstring + ')', f.func_globals)
    else:
        return eval('lambda ' + varstring + ': f(' + shortvarstring + ')', {'f':f})

def check_equations_units(eqs, x):
    '''
    Check the units of the differential equations, using
    the units of x.
    df_i/dt must have units of x_i / time.
    '''
    try:
        for f, x_i in zip(eqs, x):
            f.func_globals['xi'] = 0 * second ** -.5 # Noise
            f(*x) + (x_i / second) # Check that the two terms have the same dimension
    except DimensionMismatchError, inst:
        raise DimensionMismatchError("The differential equations are not homogeneous!", *inst._dims)

def get_var_names(eqs):
    '''
    Get the variable names from the set of equations.
    Returns a list.
    N.B.: the order is preserved.
    '''
    names = list(eqs[0].func_code.co_varnames)
    for eq in eqs:
        for name in eq.func_code.co_varnames:
            if not(name in names):
                names.append(name)
    return names


class AffineFunction(object):
    '''
    An object that can be added and multiplied by a float (or array or int).
    '''
    def __init__(self, a=1., b=0.):
        '''
        Defines an affine function as a*x+b.
        '''
        self.a = a
        self.b = b

    def __add__(self, y):
        if isinstance(y, AffineFunction):
            return AffineFunction(self.a + y.a, self.b + y.b)
        else:
            return AffineFunction(self.a, self.b + array(y))

    def __radd__(self, x):
        if isinstance(x, AffineFunction):
            return AffineFunction(self.a + x.a, self.b + x.b)
        else:
            return AffineFunction(self.a, self.b + array(x))

    def __neg__(self):
        return AffineFunction(-self.a, -self.b)

    def __sub__(self, y):
        if isinstance(y, AffineFunction):
            return AffineFunction(self.a - y.a, self.b - y.b)
        else:
            return AffineFunction(self.a, self.b - array(y))

    def __rsub__(self, x):
        if isinstance(x, AffineFunction):
            return AffineFunction(x.a - self.a, x.b - self.b)
        else:
            return AffineFunction(-self.a, array(x) - self.b)

    def __mul__(self, y):
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, array):
            return AffineFunction(self.a * array(y), self.b * array(y))
        else:
            return y.__rmul__(self)

    def __rmul__(self, x):
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, array):
            return AffineFunction(array(x) * self.a, array(x) * self.b)
        else:
            return x.__mul__(self)

    def __div__(self, y):
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, array):
            return AffineFunction(self.a / array(y), self.b / array(y))
        else:
            return y.__rdiv__(self)

    def __repr__(self):
        return str(self.a) + '*x+' + str(self.b)


class Term(object):
    '''
    A variable that can be used to isolate terms in
    a function, e.g.:
    f=lambda x:a+3*x
    f(Term()) --> 3*Term()
    Idea:
      a+x(z) = x(z) + a = x(z)
      a*x(z) = x(z)*a = x(a*z)
    '''
    def __init__(self, x=1.):
        self.x = x

    def __add__(self, y):
        return self

    def __radd__(self, y):
        return self

    def __mul__(self, y):
        return Term(self.x * y)

    def __rmul__(self, y):
        return Term(self.x * y)

    def __div__(self, y):
        return Term(self.x / y)

    def __neg__(self):
        return Term(-self.x)

    def __repr__(self):
        return str(self.x) + '*Term()'

    def __print__(self):
        return str(self.x) + '*Term()'

def is_affine(f):
    '''
    Tests whether f is an affine function.
    '''
    nargs = f.func_code.co_argcount
    try:
        f(*([AffineFunction()]*nargs))
    except:
        return False
    return True

#def is_affine1st(f,x0):
#    '''
#    Tests whether f is affine in its 1st variable.
#    '''
#    nargs=f.func_code.co_argcount
#    return is_affine(lambda x:f(x,*x0))

def depends_on(f, x, x0):
    '''
    Tests whether f depends on global variable x.
    N.B.: returns True also if f generates an error
    (e.g. with undefined variables).
    x0 is the test value for the variables (tuple).
    
    Other idea: use 'xi' in f.func_code.co_names (but not working for nested functions)
    '''
    #return x in f.func_code.co_names
#    if x in f.func_globals:
#        oldx=f.func_globals[x]
#    else:
#        oldx=None

    old_func_globals = copy(f.func_globals)

    x0 = list_replace_quantity_with_pure(x0)
    f.func_globals.update(namespace_replace_quantity_with_pure(f.func_globals))

    result = False
    f.func_globals[x] = None
    nargs = f.func_code.co_argcount
    try:
        f(*x0)
    except:
        result = True

#    if oldx==None:
#        del f.func_globals[x] # x was not defined
#    else:
#        f.func_globals[x]=oldx # previous value

    f.func_globals.update(old_func_globals)

    return result

def get_global_term(f, x, x0):
    '''
    Extract the term in global variable x from function x,
    returns a float.
    x0 is the test value for the variables (tuple).
    Example:
    getterm(lambda x:2*x+5*y,'y',(0,0))) --> 5
    '''
    old_func_globals = copy(f.func_globals)

    x0 = list_replace_quantity_with_pure(x0)
    f.func_globals.update(namespace_replace_quantity_with_pure(f.func_globals))

    f.func_globals[x] = Term()

    result = f(*x0).x

    f.func_globals.update(old_func_globals)

    return result
