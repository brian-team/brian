from codegen import *

__all__ = ['PythonCodeGenerator']


class PythonCodeGenerator(CodeGenerator):
    def initialisation(self, eqs):
        code = ''
        for i, var in enumerate(eqs._diffeq_names):
            code += var + ' = _S[' + str(i) + ']\n'
        return code

    def single_statement(self, expr):
        return CodeGenerator.single_statement(self, expr).strip()

    def single_expr(self, expr):
        return CodeGenerator.single_expr(self, expr).strip()

    def single_substitute(self, s):
#        if s==0:
#            return 'zeros(num_neurons)'
#        if s==1:
#            return 'ones(num_neurons)'
        return str(s)
