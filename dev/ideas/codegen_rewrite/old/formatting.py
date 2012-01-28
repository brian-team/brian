from string import Formatter
import re

__all__ = ['CodeFormatter', 'word_substitute']

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
