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
"""Magic function tools

Use these functions to automagically find objects of a particular class. The way it works is that
whenever a new object is created from a class derived from InstanceTracker, it is stored along
with the 'frame' it was called from (loosely speaking, the function where the object is defined).
When you call one of the functions below, it picks out all the objects of the required type in
a frame a specified number of levels before the current one (e.g. in the frame of the calling
function). 

Functions
---------

get_instances(instancetype, level=1, all=False)
    This function finds all instances at the given level in the sequence of
    calling frames.
    If all is True then levels will not be used and all instances will be found.
    
find_instances(instancetype, startlevel=1, all=False)
    This function searches the frames starting from the given startlevel until it finds at least
    one object of the required type, at which point it will return all objects of that type from
    that level
    If all is True then levels will not be used and all instances will be found.

find_all_instances(instancetype,startlevel=1, all=False):
    This function searches the frames starting from the given startlevel, and returns all objects of
    the required type from all levels. Noe that includeglobals is set to False by default so as not
    to pick up multiple copies of objects
    If all is True then levels will not be used and all instances will be found.

Variables:

instancetype
    A class (must be derived from InstanceTracker)
level
    The level in the sequence of calling frames. So, for a function f, calling with level=0 will
    find variables defined in that function f, whereas calling with level=1 will find variables
    defined in the function which called f. The latter is the default value because magic
    functions are usually used within Brian functions to find variables defined by the user.

Return values:

All the functions return a tuple (objects, names) where objects is the list of matching objects,
and names is a list of strings giving the objects' names if they are defined. At the moment, the
only name returned is the id of the object.

Notes:

These functions return each object at most once.

Classes
-------

InstanceTracker
    Derive your class from this one to automagically keep track of instances of it. If you
    want a subclass of a tracked class not to be tracked, define the method _track_instances
    to return False.
      
"""

__docformat__ = "restructuredtext en"

from weakref import *
from inspect import *
from globalprefs import *

__all__ = [ 'get_instances', 'find_instances', 'find_all_instances', 'magic_register', 'magic_return', 'InstanceTracker' ]


class ExtendedRef(ref):
    """A weak reference which also defines an optional id
    """
    def __init__(self, ob, callback=None, **annotations):
        super(ExtendedRef, self).__init__(ob, callback)
        self.__id = 0

    def set_i_d(self, id):
        self.__id = id

    def get_i_d(self):
        return self.__id


class WeakSet(set):
    """A set of extended references
    
    Removes references from the set when they are destroyed."""
    def add(self, value, id=0):
        wr = ExtendedRef(value, self.remove)
        wr.set_i_d(id)
        set.add(self, wr)

    def set_i_d(self, value, id):
        for _ in self:
            if _() is value:
                _.set_i_d(id)
                return

    def get_i_d(self, value):
        for _ in self:
            if _() is value:
                return _.get_i_d()

    def get(self, id=None):
        if id is None:
            return [ _() for _ in self if _.get_i_d() != -1 ]
        else:
            return [ _() for _ in self if _.get_i_d() == id]


class InstanceFollower(object):
    """Keep track of all instances of classes derived from InstanceTracker
    
    The variable __instancesets__ is a dictionary with keys which are class
    objects, and values which are WeakSets, so __instanceset__[cls] is a
    weak set tracking all of the instances of class cls (or a subclass).
    """
    __instancesets__ = {}
    def add(self, value, id=0):
        for cls in value.__class__.__mro__: # MRO is the Method Resolution Order which contains all the superclasses of a class
            if cls not in self.__instancesets__:
                self.__instancesets__[cls] = WeakSet()
            self.__instancesets__[cls].add(value, id)

    def set_i_d(self, value, id):
        for cls in value.__class__.__mro__: # MRO is the Method Resolution Order which contains all the superclasses of a class
            if cls in self.__instancesets__:
                self.__instancesets__[cls].set_i_d(value, id)

    def get_i_d(self, value):
        for cls in value.__class__.__mro__:
            if cls in self.__instancesets__:
                return self.__instancesets__[cls].get_i_d(value)

    def get(self, cls, id=None):
        if not cls in self.__instancesets__: return []
        return self.__instancesets__[cls].get(id)


class InstanceTracker(object):
    """Base class for all classes whose instances are to be tracked
    
    Derive your class from this one to automagically keep track of instances of it. If you
    want a subclass of a tracked class not to be tracked, define the method _track_instances
    to return False.
    """
    __instancefollower__ = InstanceFollower() # static property of all objects of class derived from InstanceTracker
    @staticmethod
    def _track_instances():
        return True

    def set_instance_id(self, idvalue=None, level=1):
        if idvalue is None:
            idvalue = id(getouterframes(currentframe())[level + 1][0])
        self.__instancefollower__.set_i_d(self, idvalue)

    def get_instance_id(self):
        return self.__instancefollower__.get_i_d(self)

    def __new__(typ, *args, **kw):
        obj = object.__new__(typ)#, *args, **kw)
        outer_frame = id(getouterframes(currentframe())[1][0]) # the id is the id of the calling frame
        if obj._track_instances():
            obj.__instancefollower__.add(obj, outer_frame)
        return obj

def magic_register(*args, **kwds):
    '''Declare that a magically tracked object should be put in a particular frame
    
    **Standard usage**
    
    If ``A`` is a tracked class (derived from :class:`InstanceTracker`), then the following wouldn't
    work::
    
        def f():
            x = A('x')
            return x
        objs = f()
        print get_instances(A,0)[0]
    
    Instead you write::
    
        def f():
            x = A('x')
            magic_register(x)
            return x    
        objs = f()
        print get_instances(A,0)[0]
    
    **Definition**
    
    Call as::
    
        magic_register(...[,level=1])
    
    The ``...`` can be any sequence of tracked objects or containers of tracked objects,
    and each tracked object will have its instance id (the execution frame in which it was
    created) set to that of its parent (or to its parent at the given level). This is
    equivalent to calling::
    
        x.set_instance_id(level=level)
    
    For each object ``x`` passed to :func:`magic_register`.
    '''
    level = kwds.get('level', 1)
    for x in args:
        if isinstance(x, InstanceTracker):
            x.set_instance_id(level=level + 1)
        else:
            magic_register(*x, **{'level':level + 1})


def magic_return(f):
    '''
    Decorator to ensure that the returned object from a function is recognised by magic functions
    
    **Usage example:** ::

        @magic_return
        def f():
            return PulsePacket(50*ms, 100, 10*ms)
    
    **Explanation**
    
    Normally, code like the following wouldn't work::

        def f():
            return PulsePacket(50*ms, 100, 10*ms)
        pp = f()
        M = SpikeMonitor(pp)
        run(100*ms)
        raster_plot()
        show()
    
    The reason is that the magic function :func:`run()` only recognises objects created
    in the same execution frame that it is run from. The :func:`magic_return` decorator
    corrects this, it registers the return value of a function with the magic
    module. The following code will work as expected::

        @magic_return
        def f():
            return PulsePacket(50*ms, 100, 10*ms)
        pp = f()
        M = SpikeMonitor(pp)
        run(100*ms)
        raster_plot()
        show()
    
    **Technical details**
    
    The :func:`magic_return` function uses :func:`magic_register` with the default ``level=1``
    on just the object returned by a function. See details for :func:`magic_register`.
    '''
    def new_f(*args, **kwds):
        obj = f(*args, **kwds)
        magic_register(obj)
        return obj
    new_f.__name__ = f.__name__
    new_f.__doc__ = f.__doc__
    return new_f

def get_instances(instancetype, level=1, all=False):
    """Find all instances of a given class at a given level in the stack
    
    See documentation for module Brian.magic
    """
    try:
        instancetype.__instancefollower__
    except AttributeError:
        raise InstanceTrackerError('Cannot track instances of type ', instancetype)
    target_frame = id(getouterframes(currentframe())[level + 1][0])
    if all or not get_global_preference('magic_useframes'):
        target_frame = None
    objs = instancetype.__instancefollower__.get(instancetype, target_frame)
    return (objs, map(str, map(id, objs)))

def find_instances(instancetype, startlevel=1, all=False):
    """Find first instances of a given class in the stack
    
    See documentation for module Brian.magic
    """
    # Note that we start from startlevel+1 because startlevel means from the calling function's point of view 
    for level in range(startlevel + 1, len(getouterframes(currentframe()))):
        objs, names = get_instances(instancetype, level, all=all)
        if len(objs):
            return (objs, names)
    return ([], [])

def find_all_instances(instancetype, startlevel=1, all=False):
    """Find all instances of a given class in the stack
    
    See documentation for module Brian.magic
    """
    objs = []
    names = []
    # Note that we start from startlevel+1 because startlevel means from the calling function's point of view 
    for level in range(startlevel + 1, len(getouterframes(currentframe()))):
        newobjs, newnames = get_instances(instancetype, level, all=all)
        objs += newobjs
        names += newnames
    return (objs, names)

if __name__ == '__main__':
    print __doc__
