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
Optimizations for Brian.
'''
from scipy.weave import blitz
import re
import warnings
import parser
from inspection import *
from log import *
try:
    import sympy
    use_sympy = True
except:
    warnings.warn('sympy not installed')
    use_sympy = False
#TODO: also insert a global pref?

__all__ = ['freeze', 'simplify_expr', 'symbolic_eval']

def freeze(expr, vars, namespace={}, safe=False):
    """
    Replaces all identifiers in expr by their float value.
    The variables vars are not changed.
    If safe is True, freezing fails if one variable is not a quantity
    """
    # Find variables
    ids = [name for name in get_identifiers(expr) if name not in vars]
    # Check that they are in the namespaces and find their value
    value = {}
    for id in ids:
        if id in namespace:
            value[id] = namespace[id]
        else:
            log_warn('brian.optimizer.freeze', "Freezing impossible because the value of " + id + " is missing")
            return None
        if not isinstance(value[id], (int, float)): # or unit?
            if safe:
                log_warn('brian.optimizer.freeze', "Freezing impossible because " + id + " is not a number")
                return None
            else:
                value[id] = id
        else:
            value[id] = float(value[id]) # downcast Quantity to float
    # Substitute
    for id in ids:
        if isinstance(value[id], float):
            strver = repr(value[id])
        else:
            strver = str(value[id])
        expr = re.sub("\\b" + id + "\\b", strver, expr)
    # Clean (changes -- to +)
    expr = re.sub("--", "+", expr)
    #print "freezing:",expr
    #return simplify_expr(expr)
    return expr

def symbolic_eval(expr):
    """
    Evaluates expr as a symbolic expression.
    """
    if not use_sympy:
        return expr
    # TODO: not with all symbols
    # Find all symbols
    namespace = {}
    vars = get_identifiers(expr)
    for var in vars:
        namespace[var] = sympy.Symbol(var)
    return eval(expr, namespace)

def simplify_expr(expr):
    '''
    Simplifies a string expression for an equation.
    NB: does not seem to yield any speed up!
    '''
    return str(symbolic_eval(expr))
    #return str(sympy.simplify(symbolic_eval(expr)))
