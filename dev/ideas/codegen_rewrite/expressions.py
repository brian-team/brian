from brian import *
from sympy.printing.ccode import CCodePrinter
import sympy
import parser
from formatting import *
    
languages = ['Python', 'C', 'GPU']

def get_identifiers(expr):
    return parser.suite(expr).compile().co_names


def symbolic_eval(expr):
    # Find all symbols
    namespace = {}
    vars = get_identifiers(expr)
    for var in vars:
        namespace[var] = sympy.Symbol(var)
    return eval(expr, namespace)


class Expression(object):
    def __init__(self, expr):
        self.expr = expr
        self.sympy_expr = symbolic_eval(self.expr)
        
    def __str__(self):
        return self.expr
    
    def substitute_symbols(self, expr, symbols):
        substitutions = dict((name, sym.read) for name, sym in symbols.iteritems())
        return word_substitute(expr, substitutions)
        
    def convert_to(self, language, symbols={}):
        if language.name=='python':
            return self.substitute_symbols(self.expr, symbols)
        elif language.name=='c':
            return self.substitute_symbols(
                CCodePrinter().doprint(self.sympy_expr),
                symbols)    


class Statement(object):
    def __init__(self, var, op, expr, boolean=False):
        self.var = var.strip()
        self.op = op.strip()
        self.expr = expr
        self.boolean = boolean
        
    def __str__(self):
        return self.var+' '+self.op+' '+str(self.expr)
            
    def convert_to(self, language, symbols={}):
        if self.var in symbols:
            sym = symbols[self.var]
            if self.op==':=':
                initial = sym.define+' = '
            else:
                initial = sym.write+' '+self.op+' '
        else:
            if self.op==':=':
                if language.name=='python':
                    initial = self.var+' = '
                elif language.name=='c':
                    if self.boolean:
                        initial = 'bool '+self.var+' = '
                    else:
                        initial = language.scalar+' '+self.var+' = '
            else:
                initial = self.var+' '+self.op+' '
        statementstr = initial+self.expr.convert_to(language, symbols)
        if language.name=='c':
            statementstr += ';'
        return statementstr
    
    
if __name__=='__main__':
    stmt = Statement('v', '=', Expression('x**2+v*((v>10)&(v<20))'))
    print 'Python:', stmt.convert_to_python()
    print 'C:', stmt.convert_to('C')
    print 'GPU:', stmt.convert_to('GPU')
