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
    '''
    For each symbol which appears in the dict ``symbols`` and as a word in the
    expression string ``expr``, replace the symbol with ``sym.read()``.
    '''
    substitutions = dict((name, sym.read()) for name, sym in symbols.iteritems())
    return word_substitute(expr, substitutions)

class Expression(object):
    '''
    A mathematical expression such as ``x*y+z``.
    
    Has an attribute ``dependencies`` which is ``Read(var)`` for all words
    ``var`` in ``expr``.
    
    Has a method :meth:`convert_to` defined the same way as
    :meth:`CodeItem.convert_to`.
    '''
    def __init__(self, expr):
        self.expr = expr
        self.sympy_expr = symbolic_eval(self.expr)
        self.dependencies = set(Read(x) for x in get_identifiers(expr))
        self.resolved = set()
        
    def __str__(self):
        return self.expr
            
    def convert_to(self, language, symbols={}, namespace={}):
        '''
        Converts expression into a string for the given ``language`` using the
        given set of ``symbols``. Replaces each :class:`Symbol` appearing in the
        expression with ``sym.read()``, and if the language is C++ or GPU then
        uses ``sympy.CCodePrinter().doprint()`` to convert the syntax, e.g.
        ``x**y`` becomes ``pow(x,y)``.
        '''
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
