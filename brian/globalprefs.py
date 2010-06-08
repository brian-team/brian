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
Global preferences for Brian
----------------------------

The following global preferences have been defined:

"""
# the global preferences referred to above are automatically
# added to the docstring when they are defined via
# define_global_preference
__docformat__ = "restructuredtext en"

__all__ = ['set_global_preferences', 'get_global_preference', 'exists_global_preference', 'define_global_preference']

import sys
from utils.documentation import *

globalprefdocs = ""


class BrianGlobalPreferences:
    pass
g_prefs = BrianGlobalPreferences()


def set_global_preferences(**kwds):
    """Set global preferences for Brian
    
    Usage::
    
        ``set_global_preferences(...)``
    
    where ... is a list of keyword assignments.
    """
    for k, v in kwds.iteritems():
        g_prefs.__dict__[k] = v


def get_global_preference(k):
    """Get the value of the named global preference
    """
    return g_prefs.__dict__[k]

def exists_global_preference(k):
    """Determine if named global preference exists
    """
    return hasattr(g_prefs, k)

def define_global_preference(k, defaultvaluedesc, desc):
    """Define documentation for a new global preference
    
    Arguments:
    
    ``k``
        The name of the preference (a string)
    ``defaultvaluedesc``
        A string description of the default value
    ``desc``
        A multiline description of the preference in
        docstring format
    """
    global globalprefdocs
    globalprefdocs += '``' + k + ' = ' + defaultvaluedesc + '``\n'
    globalprefdocs += flattened_docstring(desc, numtabs=1)
