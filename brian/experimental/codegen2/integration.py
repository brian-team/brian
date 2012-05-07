from brian import *
from expressions import *
from formatting import *
from equations import *
from statements import *

__all__ = ['EquationsContainer', 'make_integration_step',
           'euler',
           'rk2',
           'exp_euler',
           ]

class EquationsContainer(object):
    '''
    Utility class for defining numerical integration scheme
    
    Initialise with a set of equations ``eqs``. You can now iterate over
    this object in two ways, firstly over all the differential equations::
    
        for var, expr in eqscontainer:
            yield f(expr)
    
    Or over just the differential equations with nonzero expressions (i.e.
    not including ``dx/dt=0`` for parameters)::        
    
        for var, expr in eqscontainer.nonzero:
            yield f(expr)
    
    Here ``var`` is the name of the symbol, and ``expr`` is a string, the
    right hand side of the differential equation ``dvar/dt=expr``.
         
    Also has attributes:
    
    ``names``
        The symbol names for all the differential equations
    ``names_nonzero``
        The symbol names for all the nonzero differential equations
    '''
    def __init__(self, eqs):
        frozen = self.frozen = frozen_equations(eqs)
        self.all = [(var, frozen[var]) for var in eqs._diffeq_names]
        self.nonzero = [(var, frozen[var]) for var in eqs._diffeq_names_nonzero]
        self.names = eqs._diffeq_names
        self.names_nonzero = eqs._diffeq_names_nonzero
    def __iter__(self):
        return iter(self.all)

def make_integration_step(method, eqs):
    '''
    Return an integration step from a method and a set of equations.
    
    The ``method`` should be a function ``method(eqs)`` which receives a
    :class:`EquationsContainer` object as its argument, and ``yield`` s
    statements. For example, the :func:`euler` integration step is defined as::

        def euler(eqs):
            for var, expr in eqs.nonzero:
                yield '_temp_{var} := {expr}'.format(var=var, expr=expr)
            for var, expr in eqs.nonzero:
                yield '{var} += _temp_{var}*dt'.format(var=var, expr=expr)
    '''
    eqs_container = EquationsContainer(eqs)
    statements = []
    for s in method(eqs_container):
        statements.extend(statements_from_codestring(s))
    return statements

def euler(eqs):
    '''
    Euler integration
    '''
    for var, expr in eqs.nonzero:
        yield '_temp_{var} := {expr}'.format(var=var, expr=expr)
    for var, expr in eqs.nonzero:
        yield '{var} += _temp_{var}*dt'.format(var=var, expr=expr)

def rk2(eqs):
    '''
    2nd order Runge-Kutta integration
    '''
    for var, expr in eqs:
        yield '''
            _buf_{var} := {expr}
            _half_{var} := .5*dt*_buf_{var}
            _half_{var} += {var}
            '''.format(var=var, expr=expr)
    for var, expr in eqs.nonzero:
        half_subst = dict((var, '_half_'+var) for var in eqs.names)
        expr = word_substitute(expr, half_subst)
        yield '''
            _buf_{var} = {expr}
            {var} += dt*_buf_{var}
            '''.format(var=var, expr=expr)

def exp_euler(eqs):
    '''
    Exponential-Euler integration
    '''
    for var, expr in eqs.nonzero:
        subs = {
            'expr_B': word_substitute(expr, {var:0}),
            'expr_A': word_substitute(expr, {var:1}),
            'var_B': '_B_'+var,
            'var_A': '_A_'+var,
            }
        yield '''
            {var_B} := {expr_B}
            {var_A} := {expr_A}
            {var_A} -= {var_B}
            {var_B} /= {var_A}
            {var_A} *= dt
            '''.format(**subs)
    for var, expr in eqs.nonzero:
        yield '''
            {var} += _B_{var}
            {var} *= exp(_A_{var})
            {var} -= _B_{var}
            '''.format(var=var)

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
    for stmt in make_integration_step(rk2, eqs):
        print str(stmt)
