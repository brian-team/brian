#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


test_module = Extension('_test',
                           sources=['test_wrap.c', 'test.c'],
                           )

setup (name = 'test',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [test_module],
       py_modules = ["test"],
       )