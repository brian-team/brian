'''

'''
from brian import *
from dependencies import *
from formatting import *
from codeobject import *
from gpu import *

__all__ = ['CodeItem']

class CodeItem(object):
    '''
    An item of code, can be anything from a single statement corresponding to
    a single line of code, right up to a block with nested loops, etc.
    
    Should define the following attributes (default values are provided):
    
    ``resolved``
        The set of dependencies which have been resolved in this item, including
        in items contained within this item. Default value: the union of
        ``selfresolved`` and ``subresolved``. Elements of the set should be
        of type :class:`Dependency` (i.e. :class:`Read` or :class:`Write`).
    ``selfresolved``
        The set of dependencies resolved only in this item, and not in subitems.
        Default value: ``set()``.
    ``subresolved``
        The set of dependencies resolved in subitems, default value is the
        union of ``item.dependencies`` for each ``item`` in this item.
        Requires the :class:`CodeItem` to have an iterator, i.e. a method
        ``__iter__``.
    ``dependencies``, ``selfdependencies``, ``subdependencies``
        As above for resolved, but giving the set of dependencies in this code.
        The default value for ``dependencies`` takes the union of
        ``selfdependencies`` and ``subdependencies`` and removes all the
        symbols in ``resolved``.
        
    This structure of having default implementations allows several routes to
    derive a class from here, e.g.:
    
    :class:`Block`
        Simply defines a list attribute ``contents`` which is a sequence of
        items, and implements ``__iter__`` to return ``iter(contents)``.
    :class:`CodeStatement`
        Defines a fixed string which is not language-invariant, and a fixed
        set of dependencies and resolved. The :meth:`convert_to` method simply
        returns the fixed string. Does not define an ``__iter__`` method because
        the default values for ``dependencies`` and ``resolved`` are
        overwritten.
    '''
    # Some default values to simplify coding, a class deriving from this can
    # either define selfdependencies/selfresolved or dependencies/resolved.
    # If they define only the self* ones, they also need to define an
    # iterator of contained code items.
    @property
    def subdependencies(self):
        try:
            deps = set()
            for item in self:
                deps.update(item.dependencies)
            return deps
        except Exception, e:
            print e
            raise
    
    @property
    def subresolved(self):
        try:
            res = set()
            for item in self:
                res.update(item.resolved)
            return res
        except Exception, e:
            print e
            raise
    
    def __getattr__(self, name):
        '''
        Defines some default values for resolved, dependencies,
        selfdependencies, selfresolved.
        '''
        if name=='resolved':
            return self.selfresolved.union(self.subresolved)
        elif name=='dependencies':
            deps = self.selfdependencies.union(self.subdependencies)
            for name in self.resolved:
                deps.discard(Read(name))
                deps.discard(Write(name))
            return deps
        elif name=='selfdependencies':
            return set()
        elif name=='selfresolved':
            return set()
        raise AttributeError(name)
    
    def __iter__(self):
        return NotImplemented
    
    def convert_to(self, language, symbols={}, namespace={}):
        '''
        Returns a string representation of the code for this item in the given
        language. From the user point of view, you should call
        :meth:`generate`, but in developing new :class:`CodeItem` derived
        classes you need to implement this. The default behaviour is simply
        to concatenate the strings returned by the subitems.
        '''
        s = '\n'.join(item.convert_to(language,
                                      symbols=symbols,
                                      namespace=namespace) for item in self)
        return strip_empty_lines(s)

    def generate(self, name, language, symbols, namespace=None):
        '''
        Returns a :class:`Code` object. The method resolves the symbols using
        :func:`resolve`, converts to a string with :meth:`convert_to` and then
        converts that to a :class:`Code` object with
        :meth:`Language.code_object`.
        '''
        from resolution import resolve
        block, namespace = resolve(self, symbols, namespace=namespace)
        codestr = block.convert_to(language, symbols, namespace=namespace)
        code = language.code_object(name, codestr, namespace)
        return code
