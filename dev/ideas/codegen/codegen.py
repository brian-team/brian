from brian import *
import brian.optimiser as optimiser
from string import Template

euler_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__tmp = $var_expr
        '''),
    (('foreachvar', 'all'),
        '''
        $var += ${var}__tmp
        ''')
    ]

rk2_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__buf = $var_expr
        $vartype ${var}__half = $var+dt*${var}__buf
        '''),
    (('foreachvar', 'all'),
        '''
        ${var}__buf = @substitute(var_expr, {var:var+'__buf'})
        $var += dt*${var}__buf
        ''')
    ]

exp_euler_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__B = @substitute(var_expr, {var:0})
        $vartype ${var}__A = @substitute(var_expr, {var:1})-${var}__B
        ${var}__B /= ${var}__A
        '''),
    (('foreachvar', 'all'),
        '''
        $var = ($var+${var}__B)*exp(${var}__A*dt)-${var}__B
        ''')
    ] 

class CodeGenerator(object):
    def single_expr(self, expr):
        return expr
    def vartype(self):
        return ''
    def initialisation(self, eqs):
        return ''
    def generate(self, eqs, scheme):
        code = self.initialisation(eqs)
        code += self.scheme(eqs, scheme)
        return code
    def scheme(self, eqs, scheme):
        code = ''
        all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
        vartype = self.vartype()
        for block_specifier, block_code in scheme:
            # for the moment, ignore block_specifier as it is always foreachvar all
            for line in block_code.split('\n'):
                line = line.strip()
                if line:
                    if '@' in line:
                        pass # handle this later
                    t = Template(line)
                    for var in eqs._diffeq_names:
                        namespace = eqs._namespace[var]
                        var_expr = optimiser.freeze(eqs._string[var], all_variables, namespace)
                        substitutions = {'vartype':vartype,
                                         'var':var,
                                         'var_expr':var_expr}
                        code += self.single_expr(t.substitute(**substitutions))+'\n'
        return code

class PythonCodeGenerator(CodeGenerator):
    def initialisation(self, eqs):
        return ', '.join(eqs._diffeq_names) + ' = P._S\n'
    def single_expr(self, expr):
        return expr.strip()

class CCodeGenerator(CodeGenerator):
    def generate(self, eqs, scheme):
        vartype = self.vartype()
        code = self.initialisation(eqs)
        for j, name in enumerate(eqs._diffeq_names):
            code += vartype+' *' + name + '__Sbase = S+'+str(j)+'*num_neurons;\n'
        code += 'for(int i=0;i<n;i++){\n'
        for j, name in enumerate(eqs._diffeq_names):
            code += '    double &'+name+' = *'+name+'__Sbase++;\n'
        for line in self.scheme(eqs, scheme).split('\n'):
            line = line.strip()
            if line:
                code += '    '+line+'\n'
        code += '}\n'
        return code            
    def __init__(self, dtype='double'):
        self._dtype = dtype
    def vartype(self):
        return self._dtype

if __name__=='__main__':
    eqs = Equations('''
    dV/dt = -W/(10*second) : volt 
    dW/dt = -V/(1*second) : volt
    ''')
    scheme = rk2_scheme
    print PythonCodeGenerator().generate(eqs, scheme)
    print CCodeGenerator().generate(eqs, scheme)
