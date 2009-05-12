from brian import *
import brian.optimiser as optimiser
from string import Template
import re

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
    def finalisation(self, eqs):
        return ''
    def generate(self, eqs, scheme):
        code = self.initialisation(eqs)
        code += self.scheme(eqs, scheme)
        code += self.finalisation(eqs)
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
                    origline = line
                    for var in eqs._diffeq_names:
                        line = origline
                        namespace = eqs._namespace[var]
                        var_expr = optimiser.freeze(eqs._string[var], all_variables, namespace)
                        while 1:
                            m = re.search(r'\@(\w+)\(', line)
                            if not m:
                                break
                            methname = m.group(1)
                            start, end = m.span()
                            numopen = 1
                            for i in xrange(end, len(line)):
                                if line[i]=='(':
                                    numopen += 1
                                if line[i]==')':
                                    numopen -= 1
                                if numopen==0:
                                    break
                            if numopen!=0:
                                raise SyntaxError('Parentheses unmatching.')
                            args = line[start+1:i+1]
                            exec 'line = line[:start]+self.'+args+'+line[i+1:]'
                        substitutions = {'vartype':vartype,
                                         'var':var,
                                         'var_expr':var_expr}
                        t = Template(line)
                        code += self.single_expr(t.substitute(**substitutions))+'\n'
        return code
    def single_substitute(self, s):
        return str(s)
    def substitute(self, var_expr, substitutions):
        for var, replace_var in substitutions.iteritems():
            var_expr = re.sub(r'\b'+var+r'\b', self.single_substitute(replace_var), var_expr)
        return var_expr

class PythonCodeGenerator(CodeGenerator):
    def initialisation(self, eqs):
        return ', '.join(eqs._diffeq_names) + ' = P._S\n'
    def single_expr(self, expr):
        return expr.strip()
    def single_substitute(self, s):
        if s==0:
            return 'zeros(num_neurons)'
        if s==1:
            return 'ones(num_neurons)'

class CCodeGenerator(CodeGenerator):
    def __init__(self, dtype='double'):
        self._dtype = dtype
    def vartype(self):
        return self._dtype
    def initialisation(self, eqs):
        vartype = self.vartype()
        code = ''
        for j, name in enumerate(eqs._diffeq_names):
            code += vartype+' *' + name + '__Sbase = S+'+str(j)+'*num_neurons;\n'
        return code
    def generate(self, eqs, scheme):
        vartype = self.vartype()
        code = self.initialisation(eqs)
        code += 'for(int i=0;i<n;i++){\n'
        for j, name in enumerate(eqs._diffeq_names):
            code += '    '+vartype+' &'+name+' = *'+name+'__Sbase++;\n'
        for line in self.scheme(eqs, scheme).split('\n'):
            line = line.strip()
            if line:
                code += '    '+line+'\n'
        code += '}\n'
        code += self.finalisation(eqs)
        return code            

class GPUCodeGenerator(CCodeGenerator):
    def initialisation(self, eqs):
        vartype = self.vartype()
        code = '__global__ void stateupdate(int num_neurons, '+vartype+' t, '+vartype+' *S)\n'
        code += '{\n'
        code += '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        code += '    if(i>=num_neurons) return;\n'
        for j, name in enumerate(eqs._diffeq_names):
            code += '    '+vartype+' &' + name + ' = S[i+'+str(j)+'*num_neurons];\n'
        return code
    def finalisation(self, eqs):
        return '}\n'
    def generate(self, eqs, scheme):
        vartype = self.vartype()
        code = self.initialisation(eqs)
        for j, name in enumerate(eqs._diffeq_names):
            code += '    '+vartype+' &'+name+' = *'+name+'__Sbase++;\n'
        for line in self.scheme(eqs, scheme).split('\n'):
            line = line.strip()
            if line:
                code += '    '+line+'\n'
        code += self.finalisation(eqs)
        return code            

if __name__=='__main__':
    import pprint
    eqs = Equations('''
    dV/dt = -W*V/(10*second) : volt 
    dW/dt = -V/(1*second) : volt
    ''')
    scheme = exp_euler_scheme
    print 'Equations'
    print '========='
    print eqs
    print 'Scheme'
    print '======'
    for block_specifier, block_code in scheme:
        print block_specifier
        print block_code
    print 'Python code'
    print '==========='
    print PythonCodeGenerator().generate(eqs, scheme)
    print 'C code'
    print '======'
    print CCodeGenerator().generate(eqs, scheme)
    print 'GPU code'
    print '======'
    print GPUCodeGenerator().generate(eqs, scheme)