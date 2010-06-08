#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


brianlib_module = Extension('_brianlib',
                           sources=['brianlib_wrap.cxx', 'brianlib.cpp'],
                           include_dirs=['c:\\python25\\lib\\site-packages\\numpy\\core\\include']
                           )

setup (name='brianlib',
       version='0.1',
       author="Dan Goodman",
       description="""Simple swig example from docs""",
       ext_modules=[brianlib_module],
       py_modules=["brianlib"],
       )
