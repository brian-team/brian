'''
MODEL DEFINITION LOOKS LIKE:

    // variables (including consts possibly?)
    n.varNames.push_back(tS("V"));
    n.varNames.push_back(tS("float"));
    (for each)
    
    // compile time constants
    n.pNames.push_back(tS("gNa"));
    (for each)
    
    n.simCode = tS("code with \n");
    
Inserted into the model definition template below.
'''
from brian import *
from brian.experimental.codegen2 import *

__all__ = 'genn_model_code'

model_definition_template = '''
void prepareUserModels()
{{
    neuronModel n;
{code}
    nModels.push_back(n);
}}
'''

def genn_model_code(eqs, method, dt, values=None, scalar='double'):
    code, vars, params = make_c_integrator(eqs, method, dt, values=values,
                                           scalar=scalar,
                                           timeunit=ms, timename='T')
    print code
    print
    s = []
    for v in vars:
        s.append('n.varNames.push_back(tS("{v}"));'.format(v=v))
        s.append('n.varNames.push_back(tS("{scalar}"));'.format(scalar=scalar))
    for p in params:
        s.append('n.pNames.push_back(tS("{p}"));'.format(p=p))
    code = code.replace('\n', '\\n')
    s.append('n.simCode = tS("{code}");'.format(code=code))
    code = '\n'.join('    '+line for line in s)
    return model_definition_template.format(code=code)

if __name__=='__main__':
    eqs = '''
    dv/dt = (ge*sin(2*pi*500*Hz*t)+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/tau : volt
    dgi/dt = -gi/tau_i : volt
    tau : second
    tau_i : second
    '''
    print genn_model_code(eqs, euler, dt=0.1*ms)
