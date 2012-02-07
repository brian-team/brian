from brian import *
from codeobject import *

__all__ = ['Language',
                'PythonLanguage',
                'CLanguage',
           ]

class Language(object):
    def __init__(self, name):
        self.name = name.strip().lower()
    def code_object(self, code_str, namespace):
        return self.CodeObjectClass(code_str, namespace, language=self)

class PythonLanguage(Language):
    CodeObjectClass = PythonCode
    def __init__(self):
        Language.__init__(self, 'python')
        
class CLanguage(Language):
    CodeObjectClass = CCode
    def __init__(self, scalar='double'):
        Language.__init__(self, 'c')
        self.scalar = scalar

