#! /usr/bin/env python
from distutils.core import *
from distutils      import sysconfig
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# spikequeue extension module
cspikequeue = Extension("_cspikequeue",
                        ["spikequeue_wrap.cxx", "spikequeue.cpp"],
                        include_dirs = [numpy_include]
#                        swig_opts = ['-c++']
                        )

# setup
setup(  name        = "CSpikeQueue",
        description = "Manages spikes and such",
        author      = "Victor Benichoux",
        version     = "1.0",
        ext_modules = [cspikequeue],
        py_modules = ["cspikequeue"]
        )
