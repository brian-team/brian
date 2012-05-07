__all__ = ['Dependency', 'Read', 'Write',
           'get_read_or_write_dependencies',
           ]

class Dependency(object):
    '''
    Base class for :class:`Read` and :class:`Write` dependencies.
    
    A dependency marks that a :class:`CodeItem` depends on a given symbol.
    Each dependency has a ``name``.
    '''
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return '{cls}({name})'.format(cls=self.__class__.__name__,
                                      name=self.name)
    # we define these to make sure that set collects multiple instances with
    # the same class and name together
    def __eq__(self, other):
        return other.__class__ is self.__class__ and other.name==self.name
    def __ne__(self, other):
        return not self==other
    def __hash__(self):
        return repr(self).__hash__()

class Read(Dependency):
    '''
    Used to indicate a read dependency, i.e. the value of the symbol is read.
    '''
    pass
    
class Write(Dependency):
    '''
    Used to indicate a write dependency, i.e. the value of the symbol is
    written to.
    '''
    pass

def get_read_or_write_dependencies(dependencies):
    '''
    Returns the set of names of the variables which are either read to or
    written to in a set of dependencies.
    '''
    return set(dep.name for dep in dependencies)

if __name__=='__main__':
    x = set([Read('x'), Read('x'), Write('x'), Read('y')])
    print x
    print get_read_or_write_dependencies(x)
