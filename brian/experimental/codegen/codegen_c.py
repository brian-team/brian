from codegen import *
from rewriting import *

__all__ = ['CCodeGenerator']


class CCodeGenerator(CodeGenerator):
    def __init__(self, dtype='double', sympy_rewrite=True):
        CodeGenerator.__init__(self, sympy_rewrite=sympy_rewrite)
        self._dtype = dtype

    def vartype(self):
        return self._dtype

    def initialisation(self, eqs):
        vartype = self.vartype()
        code = ''
        for j, name in enumerate(eqs._diffeq_names):
            code += vartype + ' *' + name + '__Sbase = _S+' + str(j) + '*num_neurons;\n'
        return code

    def generate(self, eqs, scheme):
        vartype = self.vartype()
        code = self.initialisation(eqs)
        code += 'for(int _i=0;_i<num_neurons;_i++){\n'
        for j, name in enumerate(eqs._diffeq_names):
            code += '    ' + vartype + ' &' + name + ' = *' + name + '__Sbase++;\n'
        for line in self.scheme(eqs, scheme).split('\n'):
            line = line.strip()
            if line:
                code += '    ' + line + '\n'
        code += '}\n'
        code += self.finalisation(eqs)
        return code

    def single_statement(self, expr):
        return CodeGenerator.single_statement(self, expr) + ';'

    def single_expr(self, expr):
        return rewrite_to_c_expression(CodeGenerator.single_expr(self, expr.strip())).strip()
