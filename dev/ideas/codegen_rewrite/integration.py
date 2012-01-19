from brian import *
from expressions import *
from formatter import *
from equations import *

__all__ = ['EquationsContainer', 'make_integration_step',
           'euler',
           'rk2',
           'exp_euler',
           ]

class EquationsContainer(object):
    def __init__(self, eqs):
        frozen = self.frozen = frozen_equations(eqs)
        self.all = [(var, frozen[var]) for var in eqs._diffeq_names]
        self.nonzero = [(var, frozen[var]) for var in eqs._diffeq_names_nonzero]
        self.names = eqs._diffeq_names
        self.names_nonzero = eqs._diffeq_names_nonzero
    def __iter__(self):
        return iter(self.all)

def make_integration_step(method, eqs):
    eqs_container = EquationsContainer(eqs)
    statements = list(method(eqs_container))
    return statements

def euler(eqs):
    for var, expr in eqs.nonzero:
        yield Statement('_temp_'+var, ':=', expr)
    for var, expr in eqs.nonzero:
        yield Statement(var, '+=', '_temp_'+var+'*dt')

def rk2(eqs):
    for var, expr in eqs:
        yield Statement('_buf_'+var, ':=', expr)
        yield Statement('_half_'+var, ':=', '.5*dt*_buf_'+var)
        yield Statement('_half_'+var, '+=', var)
    for var, expr in eqs.nonzero:
        half_subst = dict((var, '_half_'+var) for var in eqs.names)
        expr = word_substitute(expr, half_subst)
        yield Statement('_buf_'+var, '=', expr)
        yield Statement(var, '+=', 'dt*_buf_'+var)

def exp_euler(eqs):
    for var, expr in eqs.nonzero:
        expr_B = word_substitute(expr, {var:0})
        expr_A = word_substitute(expr, {var:1})
        var_B = '_B_'+var
        var_A = '_A_'+var
        yield Statement(var_B, ':=', expr_B)
        yield Statement(var_A, ':=', expr_A)
        yield Statement(var_A, '-=', var_B)
        yield Statement(var_B, '/=', var_A)
        yield Statement(var_A, '*=', 'dt')
    for var, expr in eqs.nonzero:
        yield Statement(var, '+=', '_B_'+var)
        yield Statement(var, '*=', 'exp(_A_{var})'.format(var=var))
        yield Statement(var, '-=', '_B_'+var)


if __name__=='__main__':
    tau = 10*ms
    Vt0 = 1.0
    taut = 100*ms
    eqs = Equations('''
    dV/dt = (-V+I)/tau : 1
    dI/dt = -I/tau : 1
    dVt/dt = (Vt0-Vt)/taut : 1
    ''')
    eqs.prepare()
    for stmt in make_integration_step(exp_euler, eqs):
        print str(stmt)
    