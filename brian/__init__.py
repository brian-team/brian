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
Brian
"""

__docformat__ = "restructuredtext en"

import warnings as _warnings
from scipy import *
try:
    from pylab import *
except:
    _warnings.warn("Couldn't import pylab.")
if 'x' in globals(): del x # for some reason x is defined as 'symlog' by pylab!
if 'f' in globals(): del f

from clock import *
from connection import *
from directcontrol import *
from stateupdater import *
from monitor import *
from network import *
from neurongroup import *
from plotting import *
from reset import *
from threshold import *
from units import *
from utils.tabulate import *
from utils.statistics import *
from equations import *
from quantityarray import *
from globalprefs import *
from unitsafefunctions import *
from stdunits import *
from neuronmodel import *
from utils.parameters import *
from membrane_equations import *
from compartments import *
from log import *
from credits import *
from utils.parallelpython import *
from tests import *
from magic import *
from stdp import *
from stp import *

__version__ = '1.1.2'

#import unitsafefunctions as _usf
#import numpy as _numpy
#for _k in _usf.added_knowledge:
#    if hasattr(qarray, _k):
#        exec _k+'=_numpy.'+_k

### Define global preferences which are not defined anywhere else

#import sys as _sys
#import os as _os
#_stdout = _sys.stdout
#_sys.stdout = open(_os.devnull, "w")
#import scipy as _scipy
#try:
#    _scipy.weave.inline('int x=1;',[],
#                        compiler='msvc', verbose=0,
#                        type_converters=_scipy.weave.converters.blitz)
#    _useweave = True
#    _weavecompiler = 'msvc'
#except:
#    try:
#        _scipy.weave.inline('int x=1;',[],
#                            compiler='gcc', verbose=0,
#                            type_converters=_scipy.weave.converters.blitz)
#        _useweave = True
#        _weavecompiler = 'gcc'
#    except:
#        _useweave = False
#        _weavecompiler = 'gcc'
#_sys.stdout = _stdout

define_global_preference('useweave','False',
                           desc = """
                                  Defines whether or not functions should use inlined compiled
                                  C code where defined. Requires a compatible C++ compiler.
                                  The ``gcc`` and ``g++`` compilers are probably the easiest
                                  option (use Cygwin on Windows machines). See also the
                                  ``weavecompiler`` global preference.
                                  """)
set_global_preferences(useweave=False)
define_global_preference('weavecompiler','gcc',desc='''
        Defines the compiler to use for weave compilation. On Windows machines, installing
        Cygwin is the easiest way to get access to the gcc compiler.
        ''')
set_global_preferences(weavecompiler='gcc')

# check if we were run from a file or some other source, and set the default
# behaviour for magic functions accordingly
import inspect as _inspect
import os as _os
_of = _inspect.getouterframes(_inspect.currentframe())
if len(_of)>1 and _os.path.exists(_of[1][1]):
    _magic_useframes = True
else:
    _magic_useframes = False
define_global_preference('magic_useframes',str(_magic_useframes),
                       desc = """
                              Defines whether or not the magic functions should search
                              for objects defined only in the calling frame or if they
                              should find all objects defined in any frame. This should
                              be set to ``False`` if you are using Brian from an interactive
                              shell like IDLE or IPython where each command has its own
                              frame, otherwise set it to ``True``.
                              """) 
set_global_preferences(magic_useframes = _magic_useframes)

### Update documentation for global preferences
import globalprefs as _gp
_gp.__doc__+=_gp.globalprefdocs

try:
    import brian_global_config
except ImportError:
    pass

#__all__ = dir()
#
#import sys
#if 'epydoc' in sys.modules:
#    __all__=[]