from rewriting import *
import re

__all__ = ['single_expr', 'single_statement',
           'c_single_expr', 'c_single_statement',
           'python_single_expr', 'python_single_statement',
           'gpu_single_expr', 'gpu_single_statement',
           ]

def single_expr(expr):
    return expr.strip()

def single_statement(expr, single_expr=single_expr):
    m = re.search(r'[^><=]=', expr)
    if m:
        return expr[:m.end()] + ' ' + single_expr(expr[m.end():])
    return expr

def c_single_expr(expr):
    return rewrite_to_c_expression(single_expr(expr.strip())).strip()

def c_single_statement(expr):
    return single_statement(expr, single_expr=c_single_expr) + ';'

python_single_expr = single_expr
python_single_statement = single_statement

gpu_single_expr = c_single_expr
gpu_single_statement = c_single_statement
