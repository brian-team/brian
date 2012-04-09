"""
The basic Code object
"""
from brian import *
from brian.globalprefs import get_global_preference
from scipy import weave
from formatting import *

__all__ = ['Code',
           'PythonCode',
           'CCode',
           ]

class Code(object):
    '''
    The basic Code object used for all Python/C/GPU code generation.
    
    The Code object has the following attributes:

    ``name``
        The name of the code, should be unique. This matters particularly
        for GPU code which uses the name attribute for the kernel function
        names.
    ``code_str``
        A representation of the code in string form
    ``namespace``
        A dictionary of name/value pairs in which the code will be
        executed
    ``code_compiled``
        An optional value (can be None) consisting of some representation of the
        compiled form of the code
    ``pre_code``, ``post_code``
        Two optional Code objects which can be in the same or different
        languages, and can share partially or wholly the namespace. They are
        called (respectively) before or after the current code object is
        executed.
    ``language``
        A :class:`Language` object that stores some global settings and state
        for all code in that language.
        
    Each language (e.g. PythonCode) extends some or all of the methods:
    
    ``__init__(...)``
        Unsurprisingly used for initialising the object, should call
        Code.__init__ with all of its arguments.
    ``compile()``
        Compiles the code, if necessary. If not necessary, set the
        ``code_compiled`` value to any dummy value other than None.
    ``run()``
        Runs the compiled code in the namespace.
    
    It will usually not be necessary to override the call mechanism:
    
    ``__call__(**kwds)``
        Calls ``pre_code(**kwds)``, updates the namespace with ``kwds``,
        executes the code (calls ``self.run()``) and then calls
        ``post_code(**kwds)``.
    '''
    def __init__(self, name, code_str, namespace, pre_code=None, post_code=None,
                 language=None):
        self.name = name
        self.code_str = code_str
        self.namespace = namespace
        self.code_compiled = None
        self.pre_code = pre_code
        self.post_code = post_code
        self.language = language
    def compile(self):
        pass
    def run(self):
        pass
    def __call__(self, **kwds):
        if self.pre_code is not None:
            self.pre_code(**kwds)
        self.namespace.update(kwds)
        if self.code_compiled is None:
            self.compile()
        self.run()
        if self.post_code is not None:
            self.post_code(**kwds)


class PythonCode(Code):
    def compile(self):
        self.code_compiled = compile(self.code_str, 'PythonCode()', 'exec')
    def run(self):
        exec self.code_compiled in globals(), self.namespace


class CCode(Code):
    def __init__(self, name, code_str, namespace, pre_code=None, post_code=None,
                 language=None):
        Code.__init__(self, name, code_str, namespace, pre_code=pre_code,
                      post_code=post_code, language=language)
        self._weave_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._weave_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options')
        self.code_compiled = 0
    def run(self):
        weave.inline(self.code_str, self.namespace.keys(),
                     local_dict = self.namespace,
                     #support_code=c_support_code,
                     compiler=self._weave_compiler,
                     extra_compile_args=self._extra_compile_args)


if __name__=='__main__':
    pass