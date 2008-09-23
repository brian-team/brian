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
Module that defines the QuantityArray class
"""

from brian_unit_prefs import bup
import numpy
from numpy import *
from units import *
import unitsafefunctions
from operator import isSequenceType, isNumberType
from __builtin__ import all # note that the NumPy all function doesn't do what you'd expect
import warnings
from log import *
import weakref
from itertools import izip

__all__ = [ 'QuantityArray', 'qarray', 'has_consistent_dimensions', 'safeqarray']

def consistent(*args):
    args = [x for x in args if not isSequenceType(x)]
    if len(args)<2:
        return True
    return all(have_same_dimensions(a,b) for a,b in izip(args[:-1],args[1:]))

consistent_dimensions = consistent
has_consistent_dimensions = consistent

def unique_units(x):
    if isinstance(x,Quantity):
        return [get_unit_fast(x)]
    return [1.]

def QuantityArray(arr, units=None, copy=False, allowunitchanges=True):
    return array(arr,copy=copy)
qarray = QuantityArray
safeqarray = qarray

if not bup.use_units:
    def consistent(*args):
        return True
    consistent_dimensions = consistent
    has_consistent_dimensions = consistent