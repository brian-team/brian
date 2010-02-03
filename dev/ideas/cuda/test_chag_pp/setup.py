#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import os, numpy
numpy_base_dir = os.path.split(numpy.__file__)[0]
numpy_include_dir = os.path.join(numpy_base_dir, 'core/include')

testchagpp_module = Extension('_testchagpp',
                           sources=['testchagpp_wrap.cxx',
                                    'testchagpp.cu',
                                    ],
                           include_dirs=[numpy_include_dir],
                           extra_compile_args=['-O3']
                           )

setup (name = 'testchagpp',
       version = '0.1',
       author      = "Dan Goodman",
       description = """test chag:pp gpu library""",
       ext_modules = [testchagpp_module],
       py_modules = ["testchagpp"],
       )
