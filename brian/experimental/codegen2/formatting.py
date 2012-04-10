from brian.utils.documentation import flattened_docstring, indent_string
from string import Formatter
import re

__all__ = ['word_substitute', 'TAB',
           'flattened_docstring', 'indent_string',
           'get_identifiers',
           'strip_empty_lines',
           ]

TAB = '    '

def word_substitute(expr, substitutions):
    '''
    Applies a dict of word substitutions.
    
    The dict ``substitutions`` consists of pairs ``(word, rep)`` where each
    word ``word`` appearing in ``expr`` is replaced by ``rep``. Here a 'word'
    means anything matching the regexp ``\\bword\\b``.
    '''
    for var, replace_var in substitutions.iteritems():
        expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
    return expr

def get_identifiers(expr):
    '''
    Return all the identifiers in a given string ``expr``, that is everything
    that matches a programming language variable like expression, which is
    here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.
    '''
    return re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', expr)

def strip_empty_lines(s):
    '''
    Removes all empty lines from the multi-line string ``s``.
    '''
    return '\n'.join(line for line in s.split('\n') if line.strip())
