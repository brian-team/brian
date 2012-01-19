from brian import *

__all__ =  ['Symbol',
            'NeuronGroupStateVariableSymbol',
            'get_neuron_group_symbols',
            ]

class Symbol(object):
    '''
    Base class for code generation symbols
    '''
    def __init__(self, name, language):
        self.name = name
        self.language = language
    def update_namespace(self, namespace):
        pass
    @property
    def load(self):
        return ''
    @property
    def save(self):
        return ''
    @property
    def read(self):
        return self.name
    @property
    def write(self):
        return self.name
    @property
    def define(self):
        return self.name
    @property
    def support_code(self):
        return ''
    @property
    def depends(self):
        return []

class NeuronGroupStateVariableSymbol(Symbol):
    def __init__(self, group, varname, name, language, index_name=None):
        self.group = group
        self.varname = varname
        self.index_name = index_name
        Symbol.__init__(self, name, language)
    def update_namespace(self, namespace):
        if self.language.name=='python':
            namespace[self.name] = getattr(self.group, self.varname)
        elif self.language.name=='c':
            namespace['_arr_'+self.name] = getattr(self.group, self.varname)
    @property
    def load(self):
        if self.language.name=='python':
            return ''
        elif self.language.name=='c':
            s = '{scalar} &{name} = _arr_{name}[{index_name}];'
            return s.format(scalar=self.language.scalar,
                            name=self.name,
                            index_name=self.index_name)
    @property
    def read(self):
        if self.language.name=='python':
            return self.name
        elif self.language.name=='c':
            return self.name
    @property
    def write(self):
        if self.language.name=='python':
            return self.name+'[:]'
        elif self.language.name=='c':
            return self.name
    @property
    def define(self):
        # TODO: shouldn't calling this be an error? you shouldn't be defining
        # state variables, only reading/writing
        if self.language.name=='python':
            return self.name
        elif self.language.name=='c':
            return self.language.scalar+' '+self.name
    @property
    def depends(self):
        if self.language.name=='python':
            return []
        elif self.language.name=='c':
            return [self.index_name]

def get_neuron_group_symbols(group, language, index_name='_neuron_index'):
    eqs = group._eqs
    symbols = dict(
       (name,
        NeuronGroupStateVariableSymbol(group, name, name, language,
               index_name=index_name)) for name in eqs._diffeq_names)
    return symbols
    

if __name__=='__main__':
    from languages import *
    from formatter import *
    import re
    eqs = Equations('''
    dV/dt = -V/(10*ms) : 1
    ''')
    G = NeuronGroup(10, eqs)
    
    language = Language('C', scalar='double')
    #language = Language('Python')
    
    if language.name=='python':
        sym = NeuronGroupStateVariableSymbol(G, 'V', 'V', language)
    elif language.name=='c':
        sym = NeuronGroupStateVariableSymbol(G, 'V', 'V', language,
                                             index_name='_neuron_index')

    def substitute(expr, substitutions):
        for var, replace_var in substitutions.iteritems():
            expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
        return expr
        
    namespace = {
        't':defaultclock._t,
        'dt':defaultclock._dt,
        'ms':float(ms),
        }
    subs_namespace = {
        'sym':sym,
        'substitute':substitute,
        }
    sym.update_namespace(namespace)
    if language.name=='python':
        code_template = '''
        {sym.load}
        _temp_{sym.name} = {expr}
        {sym.write} = _temp_{sym.name}*dt
        {sym.save}
        '''
    elif language.name=='c':
        code_template = '''
        for(int _neuron_index=0; _neuron_index<_num_neurons; _neuron_index++)
        {{
            {sym.load}
            _temp_{sym.name} = {expr};
            {sym.write} = _temp_{sym.name}*dt;
            {sym.save}
        }}
        '''
    
    expr = substitute(eqs._string['V'], {'V': sym.read})
    subs_namespace['expr'] = expr
    
    fmt = CodeFormatter(subs_namespace)
    print 'Namespace:'
    for k, v in namespace.iteritems():
        print '-', k, ':', v.__class__
    print 'Code:'
    print fmt.format(code_template, subs_namespace)
    print 'Dependencies:'
    print sym.depends
    
