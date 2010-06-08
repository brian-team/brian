#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import os, numpy
numpy_base_dir = os.path.split(numpy.__file__)[0]
numpy_include_dir = os.path.join(numpy_base_dir, 'core/include')

brianlib_module = Extension('_brianlib',
                           sources=['brianlib_wrap.cxx',
                                    'circular.cpp',
                                    'monitor.cpp',
                                    'neurongroup.cpp',
                                    'network.cpp',
                                    'reset.cpp',
                                    'stateupdater.cpp',
                                    'threshold.cpp',
                                    'connection.cpp',
                                    ],
                           include_dirs=[numpy_include_dir],
                           extra_compile_args=['-O3']
                           )

setup (name='brianlib',
       version='0.1',
       author="Dan Goodman",
       description="""Simple swig example from docs""",
       ext_modules=[brianlib_module],
       py_modules=["brianlib"],
       )
