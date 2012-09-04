#!/usr/bin/env python

"""
setup.py file for C++ template container
"""

from distutils.core import setup, Extension
import os

templatecontainer_module = Extension('_templatecontainer',
                                     sources=['templatecontainer_wrap.cxx',
                                              'templatecontainer.cpp',
                                              ],
                                     )

setup (name='templatecontainer',
       version='0.1',
       author="Dan Goodman",
       description="""C++ version of template container""",
       ext_modules=[templatecontainer_module],
       py_modules=["templatecontainer"],
       )
