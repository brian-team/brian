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
"""
Classes and functions for storing and using parameters
"""

__all__ = ['attribdict', 'Parameters']

from itertools import chain
from inspect import *
import numpy


class attribdict(dict):
    """
    Dictionary that can be accessed via keys
    
    Note that attributes starting with _ won't be added to the dict but
    will instead be directly added to the object.
    
    Note that an attribdict cannot have new real attributes
    added to it unless they start with _, because they will
    be added to the dict instead. To add a real attribute,
    use object.__setattr__(obj,name,val).
    """
    def __init__(self, **kwds):
        super(attribdict, self).__init__(**kwds)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, val):
        if name in dir(self) or (len(name) and name[0] == '_'):
            dict.__setattr__(self, name, val)
            return
        self[name] = val

    def __repr__(self):
        s = 'Attributes:'
        for k, v in self.iteritems():
            s += '\n    ' + k + ' = ' + str(v)
        return s


class Parameters(attribdict):
    """
    A storage class for keeping track of parameters
    
    Example usage::
    
        p = Parameters(
            a = 5,
            b = 6,
            computed_parameters = '''
            c = a + b
            ''')
        print p.c
        p.a = 1
        print p.c
    
    The first ``print`` statement will give 11, the second gives 7.
    
    Details:
    
    Call as::
    
        p = Parameters(...)
    
    Where the ``...`` consists of a list of keyword / value pairs (like a ``dict``).
    Keywords must not start with the underscore ``_`` character. Any
    keyword that starts with ``computed_`` should be a string of valid Python statements
    that compute new values based on the given ones. Whenever a non-computed value is
    changed, the computed parameters are recomputed, in alphabetical order of their
    keyword names (so ``computed_a`` is computed before ``computed_b`` for example).
    Non-computed values can be accessed and set via ``p.x``, ``p.x=1`` for example, whereas
    computed values can only be accessed and not set. New parameters can be added
    after the :class:`Parameters` object is created, including new ``computed_*`` parameters. You
    can 'derive' a new parameters object from a given one as follows::
    
        p1 = Parameters(x=1)
        p2 = Parameters(y=2,**p1)
        print p2.x
    
    Note that changing the value of ``x`` in ``p2`` will not change the value of ``x`` in ``p1`` (this
    is a copy operation).
    """
    def __init__(self, **kwds):
        super(Parameters, self).__init__(**kwds)
        self._recompute()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            try:
                return self._computed_values[name]
            except KeyError:
                raise AttributeError

    def __setattr__(self, name, val):
        if hasattr(self, '_computed_values') and name in self._computed_values:
            raise AttributeError('Cannot set computed value')
        attribdict.__setattr__(self, name, val)
        self._recompute()

    def _recompute(self):
        cv = dict(self)
        items = self.items()
        items.sort()
        for k, v in items:
            if k[:9] == 'computed_':
                v = '\n'.join([line.strip() for line in v.split('\n')])
                exec v in numpy.__dict__, cv
        for k in cv.keys():
            if k[:1] == '_': # this is used to get rid of things like __builtins__ etc.
                cv.pop(k)
        for k in self.iterkeys():
            cv.pop(k)
        object.__setattr__(self, '_computed_values', cv)

    def ascode(self, name):
        """
        Returns a string which can be executed which gives all the parameters
        
        name is the name of the Parameters variable in the local namespace:
        
        Usage:
        
        P = Parameters(x=1)
        exec P.ascode('P')
        print x
        """
        s = ''
        allvals = dict(**self)
        allvals.update(self._computed_values)
        for k in allvals.iterkeys():
            if k[:9] != 'computed_':
                s += k + '=' + name + '.' + k + '\n'
        return s

    def get_vars(self, *vars):
        '''
        Returns a tuple of variables given their names
        
        vars can be a list of string names, or a single space separated string of names.
        '''
        vars = [v.split(' ') for v in vars]
        return tuple(getattr(self, v) for v in chain(*vars))

    def __repr__(self):
        s = 'Values:'
        for k, v in self.iteritems():
            if k[:9] != 'computed_':
                s += '\n    ' + k + ' = ' + str(v)
        if len(self._computed_values):
            s += '\n_computed values:'
            for k, v in self._computed_values.iteritems():
                s += '\n    ' + k + ' = ' + str(v)
        return s

    def __call__(self, **kwds):
        '''
        Returns a copy with specified arguments overwritten
        
        Sample usage:
        
        default_p = Parameters(x=1,y=2)
        specific_p = default_p(x=3)
        '''
        p = Parameters(**self) # pylint: disable-msg=W0621
        for k, v in kwds.iteritems():
            setattr(p, k, v)
        return p

    def __reduce__(self):
        return (_load_Parameters_from_pickle, (self.items(),))

def _load_Parameters_from_pickle(items):
    return Parameters(**dict(items))

if __name__ == "__main__":
    # turn off warning about attribute defined outside __init__
    # pylint: disable-msg=W0201
    p = Parameters(
        a=5,
        b=6,
        computed_p1='''
        c = a + b
        ''',
        computed_p2='''
        x = c**2
        ''')
    q = Parameters(
        d=100,
        computed_q='''
        e = a+d
        ''',
        **p
        )
    print p.c
    p.a = 1
    print p.c
    print p
    print
    print q
    q.a = -50
    print
    print q
    print
    try:
        q.c = 5
    except AttributeError, e:
        print e
    p.y = 6
    p.computed_p3 = '''
                    w = a*b
                    '''
    print p
    p.a = 2
    print p
