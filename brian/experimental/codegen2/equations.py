from brian import *
from brian.optimiser import freeze

__all__ = ['freeze_with_equations',
           'frozen_equations',
           ]

def freeze_with_equations(inputcode, eqs, ns):
    inputcode = inputcode.strip()
    all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
    inputcode = freeze(inputcode, all_variables, ns)
    return inputcode

def frozen_equations(eqs):
    frozen_eqs = {}
    eqs.prepare()
    all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    for var in eqs._diffeq_names:
        namespace = eqs._namespace[var]
        var_expr = freeze(eqs._string[var], all_variables, namespace)
        frozen_eqs[var] = var_expr
    return frozen_eqs
