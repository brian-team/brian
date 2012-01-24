from brian.utils.documentation import flattened_docstring, indent_string
from string import Formatter
import re

__all__ = ['CodeFormatter', 'word_substitute', 'TAB',
           'flattened_docstring', 'indent_string',
           'get_identifiers']

TAB = '    '

class CodeFormatter(Formatter):
    def __init__(self, namespace=None):
        Formatter.__init__(self)
        if namespace is None:
            namespace = {}
        self.namespace = namespace
    def get_value(self, first, args, kwargs):
        try:
            return Formatter.get_value(self, first, args, kwargs)
        except KeyError:
            return eval(first, self.namespace)

def word_substitute(expr, substitutions):
    for var, replace_var in substitutions.iteritems():
        expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
    return expr

def get_identifiers(expr):
    return re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', expr)
