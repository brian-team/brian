from ...optimiser import freeze
from string import Template
import re
from rewriting import *

__all__ = ['CodeGenerator']


class CodeGenerator(object):
    '''
    CodeGenerator base class for generating code in a variety of languages from mathematical expressions
    
    Base classes can define their own behaviour, but the following scheme is expected to be fairly
    standard.
    
    The initialiser takes at least one argument ``sympy_rewrite``, which defaults to ``True``, which
    specifies whether or not expressions should be simplified by passing them through sympy, so that
    e.g. ``V/(10*0.1)`` gets simplified to ``V*1.0`` (but not to ``V`` for some reason).
    
    There are also the following methods::
    
    .. method:: generate(eqs, scheme)
    
        Returns code in the target language for the set of equations ``eqs`` according to the
        integration scheme ``scheme``. The following schemes are included by default:
        
        * ``euler_scheme``
        * ``rk2_scheme``
        * ``exp_euler_scheme``
    
        This method may also call the :meth:`initialisation` and :meth:`finalisation` methods
        for code to include at the start and end. By default, it will call the :meth:`scheme`
        method to fill in the code in between.
        
    .. method:: initialisation(eqs)
    
        Returns initialisation code.
        
    .. method:: finalisation(eqs)
    
        Returns finalisation code.
    
    .. method:: scheme(eqs, scheme)
    
        Returns code for the given set of equations integrated according to the given scheme.
        See below for an explanation of integration schemes. In the process of generating
        code from the schemes, this method may call any of the following methods:
        :meth:`single_statement` to convert a single statement into code;
        :meth:`single_expr` to convert a single expression into code;
        :meth:`substitute` to substitute different values for specific variable names in
        given expressions; :meth:`vartype` to get the variable type specifier if necessary
        for that language.
    
    .. method:: single_statement(expr)
    
        Convert the given statement into code. By default, this looks for statements of the
        form ``A = B``, ``A += B``, etc. and applies the method :meth:`single_expr` to ``B``.
        Classes deriving from the base class typically only need to write a :meth:`single_expr`
        method and not a :meth:`single_statement` one.
    
    .. method:: single_expr(expr)
    
        Convert the given expression into code. For example, for Python you can do nothing, but
        for C, for example the expression ``A**B`` should be converted to ``pow(A, B)``, and
        this is done using sympy replacement. By default, this method will simplify the
        expression using sympy if the code generator was initialised with ``sympy_rewrite=True``.
        
    .. method:: substitute(var_expr, substitutions)
    
        Makes the given substitutions into the expression. ``substitutions`` should be a
        dictionary with keys the names of the variables to substitute, and values the
        value to substitute for that expression. By default these values are passed to
        the :meth:`single_substitute` method to check if there are any language specific
        things that need to be done to it, for example something substituting ``1`` should
        be replaced by ``ones(num_neurons)`` for Python which is vectorised. Typically
        this method shouldn't need to be rewritten for derived classes, whereas
        :meth:`single_substitute` might.
    
    .. method:: single_substitute(s)
    
        Replaces ``s``, a substitution value from :meth:`substitute`, with a language
        specific version if necessary. Should return a string.
    
    **Schemes**

    A scheme consists of a sequence of pairs ``(block_specifier, block_code)``. The
    ``block_specifier`` is currently unused, the schemes available at the moment all use
    ``('foreachvar', 'all')`` in this space - other specifiers are possible, e.g.
    ``'all_nonzero'`` instead of ``'all'``. ``block_code`` is a multi-line template string
    expressing that stage of the scheme (templating explained below). For each specifier,
    code pair, a block of code is generated in that order - multiple pairs can be used for
    separating stages in an integration scheme (i.e. do something for all variables, then
    do something else for all variables, etc.).
    
    Templating has two features, the first is standard Python templating replacements, e.g.
    the template string ``'${var}__buf'`` would be replaced by ``'x__buf'`` where the Python
    variable ``var='x'``. The available variables are:
    
    ``var``
        The current variable in the loop over each variable.
    ``var_expr``
        The right hand side of the differential equation, f(x) for the equation dx/dt=f(x).
    ``vartype``
        The data type (or ``''`` if none). Should be used at the start of the statement if
        the statement is declaring a new variable.
    
    The other feature of templating is that expressions of the form ``'@func(args)'`` will
    call the method ``func`` of the code generator with the given ``args`` and be replaced by
    that expression. Typically, this is used for the :meth:`substitute` method.
    
    Example scheme::

        rk2_scheme = [
            (('foreachvar', 'all'),
                """
                $vartype ${var}__buf = $var_expr
                $vartype ${var}__half = $var+dt*${var}__buf
                """),
            (('foreachvar', 'all'),
                """
                ${var}__buf = @substitute(var_expr, {var:var+'__buf'})
                $var += dt*${var}__buf
                """)
            ]
    '''
    def __init__(self, sympy_rewrite=True):
        self.sympy_rewrite = sympy_rewrite

    def single_statement(self, expr):
        m = re.search(r'[^><=]=', expr)
        if m:
            return expr[:m.end()] + ' ' + self.single_expr(expr[m.end():])
        return expr

    def single_expr(self, expr):
        if self.sympy_rewrite is not False:
            if self.sympy_rewrite is True:
                #rewriters = [floatify_numbers]
                rewriters = []
            else:
                rewriters = self.sympy_rewrite
            return sympy_rewrite(expr.strip(), rewriters)
        return expr.strip()

    def vartype(self):
        return ''

    def initialisation(self, eqs):
        return ''

    def finalisation(self, eqs):
        return ''

    def generate(self, eqs, scheme):
        code = self.initialisation(eqs)
        code += self.scheme(eqs, scheme)
        code += self.finalisation(eqs)
        return code

    def scheme(self, eqs, scheme):
        code = ''
        all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
        vartype = self.vartype()
        for block_specifier, block_code in scheme:
            # for the moment, processing of block_specifier is very crude
            if block_specifier == ('foreachvar', 'all'):
                vars_to_use = eqs._diffeq_names
            elif block_specifier == ('foreachvar', 'nonzero'):
                vars_to_use = eqs._diffeq_names_nonzero
            for line in block_code.split('\n'):
                line = line.strip()
                if line:
                    origline = line
                    for var in vars_to_use:
                        vars = eqs._diffeq_names
                        line = origline
                        namespace = eqs._namespace[var]
                        var_expr = freeze(eqs._string[var], all_variables, namespace)
                        while 1:
                            m = re.search(r'\@(\w+)\(', line)
                            if not m:
                                break
                            methname = m.group(1)
                            start, end = m.span()
                            numopen = 1
                            for i in xrange(end, len(line)):
                                if line[i] == '(':
                                    numopen += 1
                                if line[i] == ')':
                                    numopen -= 1
                                if numopen == 0:
                                    break
                            if numopen != 0:
                                raise SyntaxError('Parentheses unmatching.')
                            args = line[start + 1:i + 1]
                            #print args
                            exec 'line = line[:start]+self.' + args + '+line[i+1:]'
                        substitutions = {'vartype':vartype,
                                         'var':var,
                                         'var_expr':var_expr}
                        t = Template(line)
                        code += self.single_statement(t.substitute(**substitutions)) + '\n'
        return code

    def single_substitute(self, s):
        return str(s)

    def substitute(self, var_expr, substitutions):
        for var, replace_var in substitutions.iteritems():
            var_expr = re.sub(r'\b' + var + r'\b', self.single_substitute(replace_var), var_expr)
        return var_expr
