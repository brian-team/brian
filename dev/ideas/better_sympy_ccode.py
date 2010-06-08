from sympy.printing.ccode import CCodePrinter
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.core.basic import S

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

def newccode(expr):
    return NewCCodePrinter().doprint(expr)

if __name__ == '__main__':
    from sympy import *

    x = Symbol('x')
    y = Symbol('y')

    z = ~(x + y)

    print newccode(z)
