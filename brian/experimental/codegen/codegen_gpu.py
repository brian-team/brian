from codegen import *
from codegen_c import *
from rewriting import *

__all__ = ['GPUCodeGenerator']


class GPUCodeGenerator(CCodeGenerator):
    def initialisation(self, eqs):
        vartype = self.vartype()
        #code = '__global__ void stateupdate(int num_neurons, '+vartype+' t, '+vartype+' dt, '+vartype+' *S)\n'
        code = '__global__ void stateupdate(int num_neurons, ' + vartype + ' t, ' + vartype + ' dt, ' + ', '.join(vartype + ' *' + var + '_arr' for var in eqs._diffeq_names) + ')\n'
        code += '{\n'
        code += '    const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;\n'
        code += '    if(neuron_index>=num_neurons) return;\n'
#        for j, name in enumerate(eqs._diffeq_names):
#            code += '    '+vartype+' ' + name + ' = S[neuron_index+'+str(j)+'*num_neurons];\n'
        return code

    def finalisation(self, eqs):
        return '}\n'

    def generate(self, eqs, scheme):
        vartype = self.vartype()
        code = self.initialisation(eqs)
        for j, name in enumerate(eqs._diffeq_names):
            code += '    ' + vartype + ' ' + name + ' = ' + name + '_arr[neuron_index];\n'
        for line in self.scheme(eqs, scheme).split('\n'):
            line = line.strip()
            if line:
                if 'exp(' in line:
                    #code += '    '+line[:-1]+'*0.0+0.0000001;\n'
                    #code += '    '+line[:2]+'=0.0;\n'
                    line = line.replace('exp(', 'expf(')
                    code += '    ' + line + '\n'
                else:
                    code += '    ' + line + '\n'
        for j, name in enumerate(eqs._diffeq_names):
            if name in eqs._diffeq_names_nonzero:
                #code += 'if(isfinite('+name+')) '+name+'_arr[neuron_index] = '+name+';\n'
                code += '    ' + name + '_arr[neuron_index] = ' + name + ';\n'
                #code += '    '+name+'_arr[neuron_index] = (0.0);\n'
                #code += '    S[neuron_index+'+str(j)+'*num_neurons] = '+name+';\n'
        code += self.finalisation(eqs)
        return code
