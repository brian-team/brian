from brian import *

__all__ = ['Language',
           'PythonLanguage',
           'CLanguage',
           ]

class Language(object):
    def __init__(self, name):
        self.name = name.strip().lower()

class PythonLanguage(Language):
    def __init__(self):
        Language.__init__(self, 'python')
        
class CLanguage(Language):
    def __init__(self, scalar='double'):
        Language.__init__(self, 'c')
        self.scalar = scalar
