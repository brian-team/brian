from brian import *
from sympy.printing.ccode import CCodePrinter
import sympy
import parser
    
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
        
    def convert_to_python(self):
        return self.expr
    
    def convert_to_c(self):
        return CCodePrinter().doprint(self.sympy_expr)
    
    convert_to_gpu = convert_to_c
    
    def convert_to(self, language):
        return self.converter[language.strip().lower()](self)
    
    converter = {'python':convert_to_python,
                 'c':convert_to_c,
                 'gpu':convert_to_gpu,
                 }


class Statement(object):
    def __init__(self, var, op, expr):
        self.var = var.strip()
        self.op = op.strip()
        self.expr = expr
        
    def convert_to_python(self):
        if self.op=='=':
            inplace = '[:]'
        else:
            inplace = ''
        return self.var+inplace+' '+self.op+' '+self.expr.convert_to_python()
    
    def convert_to_c(self):
        return self.var+' '+self.op+' '+self.expr.convert_to_c()+';'
    
    convert_to_gpu = convert_to_c
    
    def convert_to(self, language):
        return self.converter[language.strip().lower()](self)
    
    converter = {'python':convert_to_python,
                 'c':convert_to_c,
                 'gpu':convert_to_gpu,
                 }
    
if __name__=='__main__':
    stmt = Statement('v', '=', Expression('x**2+v*((v>10)&(v<20))'))
    print 'Python:', stmt.convert_to_python()
    print 'C:', stmt.convert_to('C')
    print 'GPU:', stmt.convert_to('GPU')
