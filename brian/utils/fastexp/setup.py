#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import os, numpy
numpy_base_dir = os.path.split(numpy.__file__)[0]
numpy_include_dir = os.path.join(numpy_base_dir, 'core/include')

fastexp_module = Extension('_fastexp',
                           sources=['fastexp_wrap.cxx',
                                    'fastexp.cpp',
                                    'fexp.c'],
                           include_dirs=[numpy_include_dir],
                           extra_compile_args=['-O3']
                           )

setup (name='fastexp',
       version='0.1',
       author="Dan Goodman",
       description="""Simple swig example from docs""",
       ext_modules=[fastexp_module],
       py_modules=["fastexp"],
       )
