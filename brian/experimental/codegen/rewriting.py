try:
    from sympy.printing.ccode import CCodePrinter
    from sympy.printing.precedence import precedence
    import sympy
except ImportError:
    sympy = None
    CCodePrinter = object
from ...optimiser import symbolic_eval

def boolean_printer(origop, newop, start=''):
    def f(self, expr):
        PREC = precedence(expr)
        return start + newop.join(self.parenthesize(a, PREC) for a in expr.args)
    f.__name__ = '_print_' + origop
    return f


class NewCCodePrinter(CCodePrinter):
    _print_And = boolean_printer('And', '&&')
    _print_Or = boolean_printer('Or', '||')
    _print_Not = boolean_printer('Not', '', '!')

__all__ = ['rewrite_to_c_expression', 'sympy_rewrite', 'rewrite_pow', 'floatify_numbers']

if sympy is not None:
    pow = sympy.Function('pow')
else:
    pow = None

def rewrite_pow(e):
    if not len(e.args):
        return e
    newargs = tuple(rewrite_pow(a) for a in e.args)
    if isinstance(e, sympy.Pow) and e.args[1] != -1:
        return pow(*newargs)
    else:
        return e.new(*newargs)

def floatify_numbers(e):
    if not len(e.args):
        if e.is_number:
            return sympy.Number(float(e))
        return e
    newargs = tuple(floatify_numbers(a) for a in e.args)
    return e.new(*newargs)

def make_sympy_expressions(eqs):
    exprs = {}
    for name in eqs._diffeq_names + eqs._eq_names:
        exprs[name] = symbolic_eval(eqs._string[name])
    return exprs

def sympy_rewrite(s, rewriters=None):
    if rewriters is not None:
        if callable(rewriters):
            rewriters = [rewriters]
    else:
        rewriters = []
    expr = symbolic_eval(s)
    if not hasattr(expr, 'args'):
        return str(expr)
    for f in rewriters:
        expr = f(expr)
    return str(expr)

def rewrite_to_c_expression(s):
    if sympy is None:
        raise ImportError('sympy package required for code generation.')
    e = symbolic_eval(s)
    return NewCCodePrinter().doprint(e)

def generate_c_expressions(eqs):
    exprs = make_sympy_expressions(eqs)
    cexprs = {}
    for name, expr in exprs.iteritems():
        cexprs[name] = str(rewrite_pow(expr))
    return cexprs

if __name__ == '__main__':
    from brian import *
    if True:
        s = '-V**2/(10*0.001)'
        print sympy_rewrite(s)
        print sympy_rewrite(s, rewrite_pow)
        print sympy_rewrite(s, floatify_numbers)
        print sympy_rewrite(s, [rewrite_pow, floatify_numbers])
    if False:
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area
        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV
        # Time constants
        taue = 5 * ms
        taui = 10 * ms
        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV
        we = 6 * nS # excitatory synaptic weight (voltage)
        wi = 67 * nS # inhibitory synaptic weight
        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-g_na*(m*m*m)*h*(v-ENa)-g_kd*(n*n*n*n)*(v-EK))/Cm : volt 
        dm/dt = alpham*(1-m)-betam*m : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        dh/dt = alphah*(1-h)-betah*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpham = 0.32*(mV**-1)*(13*mV-v+VT)/(exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        betam = 0.28*(mV**-1)*(v-VT-40*mV)/(exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alphah = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        betah = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alphan = 0.032*(mV**-1)*(15*mV-v+VT)/(exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        betan = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')
        cexprs = generate_c_expressions(eqs)
        for k, v in cexprs.iteritems():
            print k, ':', v
    if False:
        taum = 20 * msecond
        taue = 5 * msecond
        taui = 10 * msecond
        Ee = (0. + 60.) * mvolt
        Ei = (-80. + 60.) * mvolt
        eqs = Equations('''
        dv/dt = (-v+ge*(Ee-v)+gi*(Ei-v))*(1./taum) : volt
        dge/dt = -ge*(1./taue) : 1
        dgi/dt = -gi*(1./taui) : 1 
        ''')
        cexprs = generate_c_expressions(eqs)
        for k, v in cexprs.iteritems():
            print k, ':', v
    if False:
        x = sympy.Symbol('x')
        y = x ** 2 + 3 * x
        print y
        print rewrite_pow(y)
