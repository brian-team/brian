from codegen import *
from codegen_c import *
from rewriting import *

__all__ = ['GPUCodeGenerator']

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
