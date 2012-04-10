from brian import *
from brian.optimiser import freeze

__all__ = ['freeze_with_equations',
           'frozen_equations',
           ]

def freeze_with_equations(inputcode, eqs, ns):
    '''
    Returns a frozen version of ``inputcode`` with equations and namespace.
    
    Replaces each occurrence in ``inputcode`` of a variable name in the
    namespace ``ns`` with its value if it is of int or float type. Variables
    with names in :class:`brian.Equations` ``eqs`` are not replaced, and neither
    are ``dt`` or ``t``.
    '''
    inputcode = inputcode.strip()
    all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['dt', 't']
    inputcode = freeze(inputcode, all_variables, ns)
    return inputcode

def frozen_equations(eqs):
    '''
    Returns a frozen set of equations.
    
    Each expression defining an equation is frozen as in
    :func:`freeze_with_equations`.
    '''
    frozen_eqs = {}
    eqs.prepare()
    all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['dt', 't']
    for var in eqs._diffeq_names:
        namespace = eqs._namespace[var]
        var_expr = freeze(eqs._string[var], all_variables, namespace)
        frozen_eqs[var] = var_expr
    return frozen_eqs
