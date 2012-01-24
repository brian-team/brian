from brian import *
from brian.inspection import get_identifiers
from brian.optimiser import symbolic_eval
from sympy.printing.ccode import CCodePrinter
import sympy
from formatting import *
import re

__all__ = ['Expression',
           'Statement',
           'statements_from_codestring',
           ]
    
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
        if isinstance(expr, str):
            expr = Expression(expr)
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


def statements_from_codestring(code, eqs=None, infer_definitions=False):
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    statements = []
    if eqs is None:
        defined = set()
    else:
        defined = set(eqs._diffeq_names)
    for line in lines:
        m = re.search(r'[^><=]=', line)
        if not m:
            raise ValueError("Could not extract statement from line: "+line)
        start, end = m.start(), m.end()
        op = line[start:end].strip()
        var = line[:start].strip()
        expr = line[end:].strip()
        # var should be a single word
        if len(re.findall(r'\b\w+\b', var))!=1:
            raise ValueError("LHS in statement must be single variable name, line: "+line)
        if op=='=' and infer_definitions and var not in defined:
            op = ':='
            defined.add(var)
        stmt = Statement(var, op, expr)
        statements.append(stmt)
    return statements


if __name__=='__main__':
    tau = 10*ms
    Vt0 = 1.0
    taut = 100*ms
    eqs = Equations('''
    dV/dt = (-V+I)/tau : 1
    dI/dt = -I/tau : 1
    dVt/dt = (Vt0-Vt)/taut : 1
    ''')
    reset = '''
    Vt += 0.5
    V = 0
    I = 0
    W = 5
    W = 2
    '''
    statements = statements_from_codestring(reset, eqs, infer_definitions=True)
    for stmt in statements:
        print stmt
