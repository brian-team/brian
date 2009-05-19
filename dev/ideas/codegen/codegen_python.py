from codegen import *

__all__ = ['PythonCodeGenerator']

class PythonCodeGenerator(CodeGenerator):
    def initialisation(self, eqs):
        return ', '.join(eqs._diffeq_names) + ' = _S\n'
    def single_statement(self, expr):
        return CodeGenerator.single_statement(self, expr).strip()
    def single_expr(self, expr):
        return CodeGenerator.single_expr(self, expr).strip()
    def single_substitute(self, s):
        if s==0:
            return 'zeros(num_neurons)'
        if s==1:
            return 'ones(num_neurons)'
        return str(s)
