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

model_definition_template = '''
void prepareUserModels()
{
    neuronModel n;
    {code}
    nModels.push_back(n);
}
'''

def genn_model_code(eqs, params, method, values={}):
    pass

if __name__=='__main__':
    pass
