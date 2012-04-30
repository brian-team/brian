'''
Various base classes for Brian
'''

__all__ = ['ObjectContainer']


class ObjectContainer(object):
    '''
    Implements the contained_objects protocol
    
    The object contains an attribute _contained_objects and
    a property contained_objects whose getter just returns
    _contained_objects or an empty list, and whose setter
    appends _contained_objects with the objects. This makes
    classes which set the value of contained_objects without
    thinking about things like inheritance work correctly.
    You can still directly manipulate _contained_objects
    or do something like::
    
        co = obj.contained_objects
        co[:] = []
        
    Note that when extending the list, duplicate objects are removed.
    '''
    def get_contained_objects(self):
        if hasattr(self, '_contained_objects'):
            return self._contained_objects
        self._contained_objects = []
        return self._contained_objects

    def set_contained_objects(self, newobjs):
        self._contained_objects = self.get_contained_objects()
        ids = set(id(o) for o in self._contained_objects)
        newobjs = [o for o in newobjs if id(o) not in ids]
        self._contained_objects.extend(newobjs)

    contained_objects = property(fget=get_contained_objects,
                                 fset=set_contained_objects)

if __name__ == '__main__':
    from brian import *

    class A(NetworkOperation):
        def __init__(self):
            x = NetworkOperation(lambda:None)
            print 'A:', id(x)
            self.contained_objects = [x]

    class B(A):
        def __init__(self):
            super(B, self).__init__()
            x = NetworkOperation(lambda:None)
            print 'B:', id(x)
            self.contained_objects = [x]

    b = B()
    print map(id, b.contained_objects)
