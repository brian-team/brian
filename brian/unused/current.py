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
Currents - Objects that can be plugged into membrane equations
'''
import types
from operator import isSequenceType
from stateupdater import StateUpdater
from units import *

__all__ = ['find_capacitance', 'Current', 'exp_current', 'exp_conductance', \
         'leak_current']

def find_capacitance(model):
    '''
    Tries to find the membrane capacitance from
    a set of differential equations or a model given as
    a dictionnary (with typical keys 'model', 'threshold' and 'reset').
    '''
    if type(model) == types.DictType:
        if 'Cm' in model:
            return model['Cm']
        if 'C' in model:
            return model['C']
        # Failed: look the equations
        if 'model' in model:
            model = model['model']
        else: # no clue!
            raise TypeError, 'Strange model!'

    if isinstance(model, StateUpdater):
        if hasattr(model, 'Cm'):
            return model.Cm
        if hasattr(model, 'C'):
            return model.C

    if isSequenceType(model):
        model = model[0] # The first equation is the membrane equation
    if type(model) == types.FunctionType:
        if 'Cm' in model.func_globals:
            return model.func_globals['Cm']
        if 'C' in model.func_globals:
            return model.func_globals['C']

    # Nothing was found!
    raise TypeError, 'No capacitance found!'


class Current(object):
    '''
    A membrane current.
    '''
    def __init__(self, I=lambda v:0, eqs=[]):
        '''
        I = current function, e.g. I=lambda v,g: g*(v-E)
        eqs= differential system, e.g. eqs=[lambda v,g:-g]
        '''
        self.I = I
        self.eqs = eqs

    def __radd__(self, model):
        '''
        model + self: addition of a current to a membrane equation
        '''
        Cm = find_capacitance(model)
        # Dictionnary?
        modeldict = None
        if type(model) == types.DictType:
            modeldict = model
            model = model['model']
        # Only one function?
        if type(model) == types.FunctionType:
            model = [model]
        # Assume sequence type
        n = len(model) # number of equations in model
        m = len(self.eqs) # number of equations in current
        print n, m
        newmodel = [0] * (n + m)
        # Insert current in membrane equation
        newmodel[0] = lambda * args: model[0](*(args[0:n])) + \
                    self.I(args[0], *args[n:n + m]) / Cm
        # Adjust the number of variables
        # Tricky bit here
        for i in range(1, n):
            md = model[i]
            newmodel[i] = lambda * args: md(*(args[0:n]))
        #newmodel[1:n]=[lambda *args: md(*(args[0:n])) for md in model[1:n]]
        # Add current variables
        # Adjust the number of variables
        for i in range(n, n + m):
            md = self.eqs[i - n]
            newmodel[i] = lambda * args: md(args[0], *(args[n:n + m]))
        #newmodel[n:n+m]=[lambda *args: md(args[0],*(args[n:n+m])) for md in self.eqs[n:n+m]]        
        # Dictionnary?
        if modeldict != None:
            modeldict['model'] = newmodel
            return modeldict
        else:
            return newmodel

# ---------------------
# Synaptic currents
# TODO: alpha and biexp
# ---------------------
@check_units(tau=second)
def exp_current(tau):
    '''
    Exponential current.
      I=g
      dg=-g/tau
    '''
    return Current(I=lambda v, g:g, eqs=[lambda v, g:-g / tau])

@check_units(tau=second, E=volt)
def exp_conductance(tau, E):
    '''
    Exponential conductance.
      I=g*(E-v)
      dg=-g/tau
    '''
    return Current(I=lambda v, g:g * (E - v), eqs=[lambda v, g:-g / tau])

# ------------------
# Intrinsic currents
# ------------------
@check_units(gl=siemens, El=volt)
def leak_current(gl, El):
    '''
    Leak current.
      I=gl*(El-v)
    '''
    return Current(I=lambda v:gl * (v - El), eqs=[])

# Test
if __name__ == '__main__':
    C = 2.
    dv = lambda v:-v / C
    #mymodel={'model':[dv],'Cm':3}
    m = dv + exp_conductance(10 * msecond, E= -70 * mvolt) + \
              exp_conductance(20 * msecond, E= -60 * mvolt)
    print m
    print m[0](1. * mvolt, 2., 3.), m[1](1. * volt, 2., 3.), m[2](1. * volt, 2., 3.)
