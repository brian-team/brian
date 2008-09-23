'''
Utility for Parallel Python
'''

import inspect
import os

__all__ = [ 'ppfunction' ]

brian_namespace = None

def make_brian_namespace():
    global brian_namespace
    if brian_namespace is None:
        brian_namespace = {}
        exec 'from brian import *' in brian_namespace

def ppfunction(f):
    '''
    Convenience wrapper for writing functions to use with parallelpython
    
    The parallel python module ``pp`` allows you to write code that runs simultaneously
    on multiple cores on a single machine, or multiple machines on a cluster. You can
    download the module at http://www.parallelpython.com/
    
    One annoying feature of ``pp`` is that you cannot use data values that are in the
    global namespace as part of your code, only modules and functions. This means that
    you could not execute a function with the code ``3*mV`` for example, because ``mV``
    is a data value in the :mod:`brian` namespace. Instead you would have to import
    the :mod:`brian` module and write ``3*brian.mV`` which is annoying in long chunks
    of code. This decorator simply generates a new version of your code with every
    name ``x`` from the :mod:`brian` namespace replaced by ``brian.x``. As part of this
    process, the rewritten code has to be saved to a file (because of how ``pp`` works)
    and this file is named according to the scheme
    ``basename_funcname_parallelpythonised.py``, where ``basename`` is the current
    module name and ``funcname`` is the name of the function being rewritten.
    
    **Example** ::
    
        @ppfunction
        def testf():
            return 3*mV    
    '''
    make_brian_namespace()
    source = inspect.getsource(f)
    sourcefname = inspect.getfile(f)
    source = source.replace('@ppfunction','')
    # TODO: this line could be a bit more intelligent
    names = [name for name in brian_namespace.iterkeys() if name in source]

    sourcelines = source.split('\n')
    sourcelines = [l for l in sourcelines if l.strip()]
    spaces = min([len(l)-len(l.lstrip()) for l in sourcelines])
    sourcelines = [l[spaces:] for l in sourcelines]
    fspaces = min([len(l)-len(l.lstrip()) for l in sourcelines[1:]])
    sourcelines.insert(1,' '*fspaces+'from brian import '+', '.join(names))
    sourcelines.insert(0,'from brian import *')
    source = '\n'.join(sourcelines)
    _, sourcefname = os.path.split(sourcefname)
    basefname = sourcefname[:-3]
    newbasefname = basefname+'_'+f.__name__+'_parallelpythonised'
    newfname = newbasefname + '.py'
    outfile = open(newfname,'w')
    outfile.write(source)
    outfile.write('\n')
    outfile.close()
    exec 'import ' + newbasefname
    exec 'f = ' + newbasefname + '.' + f.__name__  
    return f

if __name__=='__main__':
    @ppfunction
    def testf():
        return 3*mV
    print testf()
    print inspect.getsource(testf)    