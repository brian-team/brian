#!/usr/bin/env python

"""
setup.py file for C++ version of circular spike container
"""

from distutils.core import setup, Extension
import os, numpy
numpy_base_dir = os.path.split(numpy.__file__)[0]
numpy_include_dir = os.path.join(numpy_base_dir, 'core/include')
# 'c:\\python25\\lib\\site-packages\\numpy\\core\\include'

ccircular_module = Extension('_ccircular',
                           sources=['ccircular_wrap.cxx',
                                    'circular.cpp',
                                    ],
                           include_dirs=[numpy_include_dir],
                           )

setup (name='ccircular',
       version='0.1',
       author="Dan Goodman",
       description="""C++ version of circular spike container""",
       ext_modules=[ccircular_module],
       py_modules=["ccircular"],
       )
