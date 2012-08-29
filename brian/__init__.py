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
from connections import *
from synapses import *
from directcontrol import *
from stateupdater import *
from monitor import *
from network import *
from neurongroup import *
from plotting import *
from reset import *
from threshold import *
from units import *
from tools import *
from equations import *
from globalprefs import *
from unitsafefunctions import *
from stdunits import *
from membrane_equations import *
from compartments import *
from log import *
from magic import *
from stdp import *
from stp import *
from timedarray import *
from deprecated.multiplespikegeneratorgroup import *
from tests.simpletest import *

__version__ = '1.4.0'
__release_date__ = '2012-08-29'

### Define global preferences which are not defined anywhere else

define_global_preference(
    'useweave', 'False',
    desc="""
         Defines whether or not functions should use inlined compiled
         C code where defined. Requires a compatible C++ compiler.
         The ``gcc`` and ``g++`` compilers are probably the easiest
         option (use Cygwin on Windows machines). See also the
         ``weavecompiler`` global preference.
         """)
set_global_preferences(useweave=False)
define_global_preference(
    'weavecompiler', 'gcc',
    desc='''
         Defines the compiler to use for weave compilation. On Windows machines, installing
         Cygwin is the easiest way to get access to the gcc compiler.
         ''')
set_global_preferences(weavecompiler='gcc')

define_global_preference(
    'gcc_options', "['-ffast-math']",
    desc='''
         Defines the compiler switches passed to the gcc compiler. For gcc versions
         4.2+ we recommend using ``-march=native``. By default, the ``-ffast-math``
         optimisations are turned on - if you need IEEE guaranteed results, turn
         this switch off.
         ''')
set_global_preferences(gcc_options=['-ffast-math'])

define_global_preference(
    'usecodegen', 'False',
    desc='''
         Whether or not to use experimental code generation support.
         ''')
set_global_preferences(usecodegen=False)
define_global_preference(
    'usecodegenweave', 'False',
    desc='''
         Whether or not to use C with experimental code generation support.
         ''')
set_global_preferences(usecodegenweave=False)
define_global_preference(
    'usecodegenstateupdate', 'True',
    desc='''
         Whether or not to use experimental code generation support on state updaters.
         ''')
set_global_preferences(usecodegenstateupdate=True)
define_global_preference(
    'usecodegenreset', 'False',
    desc='''
         Whether or not to use experimental code generation support on resets.
         Typically slower due to weave overheads, so usually leave this off.
         ''')
set_global_preferences(usecodegenreset=False)
define_global_preference(
    'usecodegenthreshold', 'True',
    desc='''
         Whether or not to use experimental code generation support on thresholds.
         ''')
set_global_preferences(usecodegenthreshold=True)

define_global_preference(
    'usenewpropagate', 'False',
    desc='''
         Whether or not to use experimental new C propagation functions.
         ''')
set_global_preferences(usenewpropagate=False)

define_global_preference(
    'usecstdp', 'False',
    desc='''
         Whether or not to use experimental new C STDP.
         ''')
set_global_preferences(usecstdp=False)

define_global_preference(
    'brianhears_usegpu', 'False',
    desc='''
         Whether or not to use the GPU (if available) in Brian.hears. Support
         is experimental at the moment, and requires the PyCUDA package to be
         installed.
         ''')
set_global_preferences(brianhears_usegpu=False)

# check if we were run from a file or some other source, and set the default
# behaviour for magic functions accordingly
import inspect as _inspect
import os as _os
_of = _inspect.getouterframes(_inspect.currentframe())
if len(_of) > 1 and _os.path.exists(_of[1][1]):
    _magic_useframes = True
else:
    _magic_useframes = False
define_global_preference(
    'magic_useframes', str(_magic_useframes),
    desc="""
         Defines whether or not the magic functions should search
         for objects defined only in the calling frame or if they
         should find all objects defined in any frame. This should
         be set to ``False`` if you are using Brian from an interactive
         shell like IDLE or IPython where each command has its own
         frame, otherwise set it to ``True``.
         """)
set_global_preferences(magic_useframes=_magic_useframes)

### Update documentation for global preferences
import globalprefs as _gp
_gp.__doc__ += _gp.globalprefdocs

try:
    import brian_global_config
except ImportError:
    pass

try:
    import nose

    @nose.tools.nottest
    def run_all_tests():
        import tests
        tests.go()

except ImportError:

    def run_all_tests():
        print "Brian test framework requires 'nose' package."
