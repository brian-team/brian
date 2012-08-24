# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
"""
Functions which check the dimensions of their arguments, etc.

Functions updated to provide Quantity functionality
---------------------------------------------------

With any dimensions:

* sqrt

Dimensionless:

* log, exp
* sin, cos, tan
* arcsin, arccos, arctan
* sinh, cosh, tanh
* arcsinh, arccosh, arctanh

With homogeneous dimensions:

* dot
"""

from brian_unit_prefs import bup
from units import *
import numpy, math, scipy
from numpy import *
from numpy.random import *
from scipy.integrate import *
import inspect

__all__ = []

# these functions are the ones that will work with the template immediately below, and
# extend the numpy functions to know about Quantity objects 
quantity_versions = [
         'sqrt',
         'log', 'exp',
         'sin', 'cos', 'tan',
         'arcsin', 'arccos', 'arctan',
         'sinh', 'cosh', 'tanh',
         'arcsinh', 'arccosh', 'arctanh'
         ]

def make_quantity_version(func):
    funcname = func.__name__
    def f(x):
        if isinstance(x, Quantity):
            return getattr(x, funcname)()
        return func(x)
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    if hasattr(func, '__dict__'):
        f.__dict__.update(func.__dict__)
    return f

for name in quantity_versions:
    if bup.use_units:
        exec name + '=make_quantity_version(' + name + ')'
    __all__.append(name)
