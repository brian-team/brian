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
'''
Membrane equations for Brian models.
'''
__all__ = ['Current', 'IonicCurrent', 'InjectedCurrent', 'MembraneEquation']

from equations import *
from units import have_same_dimensions, get_unit, second, volt, amp
from warnings import warn


class Current(Equations):
    '''
    A set of equations defining a current.
    
    current_name is the name of the variable that must be added as a membrane current
    to the membrane equation.
    '''
    def __init__(self, expr='', current_name=None, level=0, surfacic=False, **kwd):
        Equations.__init__(self, expr, level=level + 1, **kwd)
        if surfacic: # A surfacic current is multiplied by membrane area in a MembraneEquation
            self._prefix = '__scurrent_'
        else:
            self._prefix = '__current_'
        # Find which variable is the current
        if current_name: # Explicitly given
            self.set_current_name(current_name)
        elif expr != '': # Guess
            if len(self._units) == 2: # only one variable (the other one is t)
                # Only 1 variable: it's the current
                correct_names = [name for name in self._units.keys() if name != 't']
                if len(correct_names) != 1:
                    raise NameError, "The equations do not include time (variable t)"
                name, = correct_names
                self.set_current_name(name)
            else:
                # Look for variables with dimensions of current: won't work with units off!
                current_names = [name for name, unit in self._units.iteritems()\
                               if have_same_dimensions(unit, amp)]
                if len(current_names) == 1: # only one current
                    self.set_current_name(current_names[0])
                else:
                    warn("The current variable could not be found!")

    def set_current_name(self, name):
        if name != 't':
            if name is None:
                name = unique_id()
            current_name = self._prefix + name
            self.add_eq(current_name, name, self._units[name]) # not an alias because read-only

    def __iadd__(self, other):
        # Adding a MembraneEquation
        if isinstance(other, MembraneEquation):
            return other.__iadd__(self)
        else:
            return Equations.__iadd__(self, other)


class IonicCurrent(Current):
    '''
    A ionic current; current direction is defined from intracellular
    to extracellular.
    '''
    def set_current_name(self, name):
        if name != 't':
            if name is None:
                name = unique_id()
            current_name = self._prefix + name
            self.add_eq(current_name, '-' + name, self._units[name])

InjectedCurrent = Current


class MembraneEquation(Equations):
    '''
    A membrane equation, defined by a capacitance C and a sum of currents.
    Ex:
      eq=MembraneEquation(200*pF)
      eq=MembraneEquation(200*pF,vm='V')
    No more than one membrane equation allowed per system of equations.
    '''
    def __init__(self, C=None, **kwd):
        if C is not None:
            expr = '''
            dvm/dt=__membrane_Im/C : volt
            __membrane_Im=0*unit : unit
            '''
            Equations.__init__(self, expr, C=C, unit=get_unit(C * volt / second), **kwd)
        else:
            Equations.__init__(self)

    def __iadd__(self, other):
        Equations.__iadd__(self, other)
        self.set_membrane_current()
        return self

    def set_membrane_current(self):
        current_vars = [name for name in self._eq_names if name.startswith('__current_')] # point current
        scurrent_vars = [name for name in self._eq_names if name.startswith('__scurrent_')] # surfacic current
        if scurrent_vars != []:
            if current_vars != []:
                self._string['__membrane_Im'] = '+'.join(current_vars) + '+__area*(' + '+'.join(scurrent_vars) + ')'
            else:
                self._string['__membrane_Im'] = '__area*(' + '+'.join(scurrent_vars) + ')'
            self._namespace[name]['__area'] = self.area
        elif current_vars != []:
            self._string['__membrane_Im'] = '+'.join(current_vars)
