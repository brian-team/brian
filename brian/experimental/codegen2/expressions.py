from brian import *
from brian.optimiser import symbolic_eval
from sympy.printing.ccode import CCodePrinter
import sympy
from formatting import *
from dependencies import *
import re

__all__ = ['Expression',
           ]

def substitute_symbols(expr, symbols):
    substitutions = dict((name, sym.read()) for name, sym in symbols.iteritems())
    return word_substitute(expr, substitutions)

class Expression(object):
    def __init__(self, expr):
        self.expr = expr
        self.sympy_expr = symbolic_eval(self.expr)
        self.dependencies = set(Read(x) for x in get_identifiers(expr))
        self.resolved = set()
        
    def __str__(self):
        return self.expr
            
    def convert_to(self, language, symbols={}, namespace={}):
        if language.name=='python':
            return substitute_symbols(self.expr, symbols)
        elif language.name=='c' or language.name=='gpu':
            return substitute_symbols(
                CCodePrinter().doprint(self.sympy_expr),
                symbols)

if __name__=='__main__':
    from languages import *
    python = PythonLanguage()
    c = CLanguage()
    expr = Expression('3*x**2+(y<z)*w')
    print expr
    print expr.convert_to(python)
    print expr.convert_to(c)
    print expr.dependencies
    print expr.resolved
    