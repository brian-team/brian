from brian import *
from codeobject import *

__all__ = ['Language',
                'PythonLanguage',
                'CLanguage',
           ]

class Language(object):
    '''
    Base class for languages, each should provide a ``name`` attribute, and a
    method :meth:`~Language.code_object`.
    '''
    def __init__(self, name):
        self.name = name.strip().lower()
    def code_object(self, name, code_str, namespace):
        '''
        Return a :class:`Code` object from a given ``name``, code string
        ``code_str`` and ``namespace``. If the class has a class attribute
        ``CodeObjectClass``, the default implementation returns::
        
            CodeObjectClass(name, code_str, namespace, language=self)
        '''
        return self.CodeObjectClass(name, code_str, namespace, language=self)

class PythonLanguage(Language):
    '''
    Python language.
    '''
    CodeObjectClass = PythonCode
    def __init__(self):
        Language.__init__(self, 'python')
        
class CLanguage(Language):
    '''
    C language.
    
    Has an attribute ``scalar='double'`` which gives the default type of
    scalar values (used when ``dtype`` is not specified). This can be used, for
    example, on the GPU where ``double`` may not be available.
    '''
    CodeObjectClass = CCode
    def __init__(self, scalar='double'):
        Language.__init__(self, 'c')
        self.scalar = scalar

