from brian import *
from brian.optimiser import symbolic_eval
from sympy.printing.ccode import CCodePrinter
import sympy
from formatting import *
from expressions import *
from codeitems import *
from dependencies import *
from languages import *
import re

__all__ = [
    'Statement',
        'CodeStatement',
            'CDefineFromArray',    
       'MathematicalStatement',
    'statements_from_codestring',
    'c_data_type',
    ]

class Statement(CodeItem):
    pass

class CodeStatement(Statement):
    def __init__(self, code, dependencies, resolved):
        self.dependencies = dependencies
        self.resolved = resolved
        if '\n' in code:
            code = flattened_docstring(code)
        self.code = code
    def __str__(self):
        return self.code
    __repr__ = __str__
    def convert_to(self, language, symbols={}, namespace={}):
        return self.code

def c_data_type(dtype):
    if dtype is None:
        dtype = float64
    if dtype is int:
        dtype = array([1]).dtype.type
    if dtype==float32:
        dtype = 'float'
    elif dtype==float64:
        dtype = 'double'
    elif dtype==int32:
        dtype = 'int32_t'
    elif dtype==int64:
        dtype = 'int64_t'
    elif dtype==bool_ or dtype is bool:
        dtype = 'bool'
    return dtype

class CDefineFromArray(CodeStatement):
    def __init__(self, var, arr, index,
                 dependencies=None, resolved=None,
                 dtype=None, reference=True, const=False):
        dtype = c_data_type(dtype)
        if reference:
            ref = '&'
        else:
            ref = ''
        if const:
            const = 'const '
        else:
            const = ''
        code = '{const}{dtype} {ref}{var} = {arr}[{index}];'.format(
            dtype=dtype, ref=ref, const=const,
            var=var, arr=arr, index=index)
        if dependencies is None:
            dependencies = set([Read(arr), Read(index)])
        if resolved is None:
            resolved = set([var])
        CodeStatement.__init__(self, code, dependencies, resolved)

class MathematicalStatement(Statement):
    def __init__(self, var, op, expr, dtype=None):
        self.var = var.strip()
        self.op = op.strip()
        if isinstance(expr, str):
            expr = Expression(expr)
        self.expr = expr
        if dtype is not None:
            dtype = c_data_type(dtype)
        self.dtype = dtype
        self.dependencies = self.expr.dependencies
        if self.op==':=':
            self.resolved = set([self.var])
        else:
            self.dependencies.add(Write(self.var))
            self.resolved = set()
        
    def __str__(self):
        return self.var+' '+self.op+' '+str(self.expr)
    __repr__ = __str__
            
    def convert_to(self, language, symbols={}, tabs=0, namespace={}):
        if self.var in symbols:
            sym = symbols[self.var]
            lhs_name = sym.write()
        else:
            lhs_name = self.var
        if self.op==':=':
            if lhs_name in namespace:
                raise ValueError("Trying to redefine existing Symbol "+self.var)
            if language.name=='python':
                initial = lhs_name+' = '
            elif language.name=='c' or language.name=='gpu':
                dtype = self.dtype
                if dtype is None:
                    dtype = language.scalar
                initial = dtype+' '+lhs_name+' = '
        else:
            initial = lhs_name+' '+self.op+' '
        statementstr = initial+self.expr.convert_to(language, symbols,
                                                    namespace=namespace)
        if language.name=='c' or language.name=='gpu':
            statementstr += ';'
        return TAB*tabs+statementstr


def statements_from_codestring(code, eqs=None, defined=None,
                               infer_definitions=False):
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    statements = []
    if defined is None:
        defined = set()
    if eqs is not None:
        defined.update(set(eqs._diffeq_names))
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
        stmt = MathematicalStatement(var, op, expr)
        statements.append(stmt)
    return statements

if __name__=='__main__':
    statements = statements_from_codestring('''
        x := y + z
        y = 7
        z += 3
        x = 2
        ''')
    for stmt in statements:
        print stmt
        print '    Deps, Resolved:', stmt.dependencies, stmt.resolved
    deps = set().union(*[stmt.dependencies for stmt in statements])
    res = set().union(*[stmt.resolved for stmt in statements])
    for r in res:
        deps.discard(Read(r))
        deps.discard(Write(r))
    print 'Deps:', deps
    print 'R/W:', get_read_or_write_dependencies(deps)
    print 'Resolved:', res
    print
    language = CLanguage()
    stmt = CDefineFromArray('V', '_arr_V', '_neuron_index', language)
    print stmt
    print 'Deps:', stmt.dependencies
    print 'Resolved:', stmt.resolved
    