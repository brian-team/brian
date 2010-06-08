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
Compartmental models for Brian.
'''
from equations import *
from membrane_equations import *
from units import check_units, ohm

#TODO: add MembraneEquation
class Compartments(Equations):
    """
    Creates a compartmental model from a dictionary
    of MembraneEquation objects. Compartments are initially unconnected.
    Caution: the original objects are modified.
    """
    def __init__(self, comps):
        Equations.__init__(self, '')
        for name, eqs in comps.iteritems():
            name = str(name)
            # Change variable names
            vars = eqs._units.keys()
            for var in vars:
                eqs.substitute(var, var + '_' + name)
            self += eqs

    @check_units(Ra=ohm)
    def connect(self, a, b, Ra):
        """
        Connects compartment a to compartment b with axial resistance Ra.
        """
        a, b = str(a), str(b)
        # Axial current from a to b
        Ia_name = 'Ia_' + a + '_' + b
        self += Equations('Ia=(va-vb)*invRa : amp', Ia=Ia_name, invRa=1. / Ra, va='vm_' + a, vb='vm_' + b)
        # Add the current to both compartments
        self._string['__membrane_Im_' + a] += '-' + Ia_name
        self._string['__membrane_Im_' + b] += '+' + Ia_name
