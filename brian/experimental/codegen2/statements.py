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
    '''
    Just a base class, supposed to indicate single line statements.
    '''
    pass

class CodeStatement(Statement):
    '''
    A language-specific single line of code, which should only be used in
    the resolution step by a :class:`Symbol` which knows the language it is
    resolving to. The string ``code`` and the set of ``dependencies`` and
    ``resolved`` have to be given explicitly.
    '''
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
    '''
    Gives the C language specifier for numpy data types. For example,
    ``numpy.int32`` maps to ``int32_t`` in C.
    
    Perhaps this method is given somewhere in numpy, but I couldn't find it.
    '''
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
    elif dtype==uint16:
        dtype = 'uint16_t'
    elif dtype==uint32:
        dtype = 'uint32_t'
    elif dtype==bool_ or dtype is bool:
        dtype = 'bool'
    return dtype

class CDefineFromArray(CodeStatement):
    '''
    Define a variable from an array and an index in C.
    
    For example::
    
        double &V = __arr_V[neuron_index];
    
    Initialisation arguments are:
    
    ``var``
        The variable being defined, a string.
    ``arr``
        A string representing the array.
    ``index``
        A string giving the index.
    ``dependencies``
        Given explicitly, or by default use ``set([Read(arr), Read(index)])``.
    ``resolved``
        Given explicitly, or by default use ``set([var])``.
    ``dtype``
        The numpy data type of the variable being defined.
    ``reference``
        Whether the variable should be treated as a C++ reference (e.g.
        ``double &V = ...`` rather than ``double V = ...``. If the variable
        is being written to as well as read from, use ``reference=True``.
    ``const``
        Whether the variable can be defined as const, specify this if only
        reading the value and not writing to it.
    '''
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
    '''
    A single line mathematical statement.
    
    The structure is ``var op expr``.
    
    ``var``
        The left hand side of the statement, the value being written to, a
        string.
    ``op``
        The operation, can be any of the standard Python operators (including
        ``+=`` etc.) or a special operator ``:=`` which means you are defining
        a new symbol (whereas ``=`` means you are setting the value of an
        existing symbol).
    ``expr``
        A string or an :class:`Expression` object, giving the right hand side
        of the statement.
    ``dtype``
        If you are defining a new variable, you need to specify its numpy dtype.
        
    If ``op==':='`` then this statement will resolve ``var``, otherwise it will
    add a :class:`Write` dependency for ``var``. The other dependencies come
    from ``expr``.
    '''
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
        '''
        When converting to a code string, the following takes place:
        
        * If the LHS variable is in the set of ``symbols``, then the LHS
          is replaced by ``sym.write()``
        * The expression is converted with :meth:`Expression.convert_to`.
        * If the operation is definition ``op==':='`` then the output is
          language dependent. For Python it is ``lhs = rhs`` and for C or
          GPU it is ``dtype lhs = rhs``.
        * If the operation is not definition, the statement is converted to
          ``lhs op rhs``.
        * If the language is C/GPU the statement has ``;`` appended.
        '''
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
    '''
    Generate a list of statements from a user-defined string.
    
    ``code``
        The input code string, a multi-line string which should be flat, no
        indents.
    ``eqs``
        A Brian :class:`~brian.Equations` object, which is used to specify
        a set of already defined variable names if you are using
        ``infer_definitions``.
    ``defined``
        A set of symbol names which are already defined, if you are using
        ``infer_definitions``.
    ``infer_definitions``
        Set to ``True`` to guess when a line of the form ``a=b`` should be
        inferred to be of type ``a:=b``, as user-specified code may not make
        the distinction between ``a=b`` and ``a:=b``.
        
    The rule for definition inference is that you scan through the lines, and
    a set of already defined symbols is maintained (starting from ``eqs`` and
    ``defined`` if provided), and an ``=`` op is changed to ``:=`` if the
    name on the LHS is not already in the set of symbols already defined.
    '''
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
