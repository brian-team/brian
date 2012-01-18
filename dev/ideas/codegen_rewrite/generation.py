from brian import *
from languages import *
from expressions import *
from formatter import *
from symbols import *
from codeobject import *
from brian.utils.documentation import flattened_docstring, indent_string
from brian.optimiser import freeze

class CodeBlock(object):
    def __init__(self, resolved, content, tabs=0):
        self.resolved = resolved
        if isinstance(content, str) and '\n' in content:
            content = [flattened_docstring(content)]
        self.content = content
        self.tabs = tabs
        
    def __str__(self):
        s = ''
        for c in self.content:
            if isinstance(c, Statement):
                c = indent_string(str(c), self.tabs)
            elif isinstance(c, str):
                c = indent_string(c, self.tabs)
            else:
                c = str(c)+'\n'
            s = s+c
        return s
    
    def indented(self, tabs=1):
        return CodeBlock(self.resolved, self.content, self.tabs+tabs)
    
    def generate_codestr(self, language, symbols, symbols_to_load=None):
        codestr = ''
        # first we generate code to load whichever variables
        # can be resolved (all dependencies met)
        if symbols_to_load is None:
            symbols_to_load = symbols
        new_symbols_to_load = {}
        for name, sym in symbols_to_load.items():
            deps = sym.depends
            if set(deps).issubset(set(self.resolved)):
                codestr += indent_string(sym.load, self.tabs)
            else:
                new_symbols_to_load[name] = sym
        # now we generate the content (recursively)
        for item in self.content:
            itemstr = None
            if isinstance(item, str):
                itemstr = indent_string(item, self.tabs)
            if isinstance(item, Statement):
                itemstr = item.convert_to(language, symbols)
                itemstr = indent_string(itemstr, self.tabs)
            if isinstance(item, CodeBlock):
                itemstr = item.generate_codestr(language, symbols,
                                                new_symbols_to_load)
            if itemstr is None:
                raise TypeError("Unknown code block item type")
            codestr = codestr+itemstr
        return codestr
    
    def generate(self, language, symbols):
        codestr = self.generate_codestr(language, symbols)
        namespace = {}
        for name, sym in symbols.iteritems():
            sym.update_namespace(namespace)
        if language.name=='python':
            code = PythonCode(codestr, namespace)
        elif language.name=='c':
            code = CCode(codestr, namespace)
        return code


def euler_integration_step(eqs, language):
    statements = []
    all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    for var in eqs._diffeq_names_nonzero:
        namespace = eqs._namespace[var]
        var_expr = freeze(eqs._string[var], all_variables, namespace)
        var_expr = Expression(var_expr)
        stmt = Statement('_temp_'+var, ':=', var_expr)
        statements.append(stmt)
    for var in eqs._diffeq_names_nonzero:
        expr = Expression('_temp_'+var+'*_dt')
        stmt = Statement(var, '+=', expr)
        statements.append(stmt)
    return statements

def make_euler_code_block(group, eqs, language):
    statements = euler_integration_step(eqs, language)
    integration_step = CodeBlock(['_neuron_index'], statements)
    if language.name=='python':
        return integration_step
    elif language.name=='c':
        loop_block = CodeBlock([], [
            CodeBlock([],
            '''
            for(int _neuron_index=0; _neuron_index<_num_neurons; _neuron_index++)
            {
            '''),
                integration_step.indented(),
            CodeBlock([],
            '}'),
            ])
        return loop_block


def make_threshold_code_block(group, threshold, language):
    if language.name=='python':
        expr = Expression(threshold)
        stmt = Statement('_spikes_bool', ':=', expr)
        return CodeBlock([], [stmt])
    elif language.name=='c':
        expr = Expression(threshold)
        stmt = Statement('_spiked', ':=', expr, boolean=True)
        threshold_step = CodeBlock(['_neuron_index'], [stmt])
        loop_block = CodeBlock([], [
            CodeBlock([],
            '''
            long int &_numspikes = _numspikes_arr[0];
            _numspikes = 0;
            for(int _neuron_index=0; _neuron_index<_num_neurons; _neuron_index++)
            {
            '''),
                threshold_step.indented(),
            CodeBlock([], '''
                if(_spiked)
                {
                    _spikes[_numspikes++] = _neuron_index;
                }
            }
            '''),
            ])
        return loop_block

if __name__=='__main__':
    tau = 10*ms
    Vt0 = 1.0
    taut = 100*ms
    eqs = Equations('''
    dV/dt = (-V+I)/tau : 1
    dI/dt = -I/tau : 1
    dVt/dt = (Vt0-Vt)/taut : 1
    ''')
    threshold = 'V>Vt'
    reset = '''
    Vt += 0.5
    V = 0
    I = 0
    '''
    G = NeuronGroup(3, eqs, threshold=threshold, reset=reset)
    G.Vt = Vt0
        
    language = CLanguage(scalar='double')
    #language = PythonLanguage()

    symbols = dict((name,
                    NeuronGroupStateVariableSymbol(G, name, name, language,
                           index_name='_neuron_index')) for name in eqs._diffeq_names)

    block = make_euler_code_block(G, eqs, language)
    code = block.generate(language, symbols)
    code.namespace['_dt'] = G.clock._dt
    code.namespace['_num_neurons'] = len(G)
    print 'State update'
    print '============'
    print 'Block:'
    print str(block)
    print 'Code string:'
    print code.code_str
    print 'Code namespace:'
    for k, v in code.namespace.iteritems():
        print '   ', k, ':', v.__class__
        
    threshold_block = make_threshold_code_block(group, threshold, language)
    threshold_code = threshold_block.generate(language, symbols)    
    threshold_code.namespace['_spikes'] = zeros(len(G), dtype=int)
    threshold_code.namespace['_numspikes_arr'] = zeros(1, dtype=int)
    threshold_code.namespace['_num_neurons'] = len(G)
    print 'Threshold'
    print '========='
    print 'Block:'
    print str(threshold_block)
    print 'Code string:'
    print threshold_code.code_str
    print 'Code namespace:'
    for k, v in threshold_code.namespace.iteritems():
        print '   ', k, ':', v.__class__

    if language.name=='python':
        def threshold_func(P):
            threshold_code()
            return threshold_code.namespace['_spikes_bool'].nonzero()[0]
    elif language.name=='c':
        def threshold_func(P):
            threshold_code()
            ns = threshold_code.namespace
            spikes = ns['_spikes'][:ns['_numspikes_arr'][0]]
            return spikes
    #exit()

    P = PoissonGroup(1, rates=300*Hz)
    C = Connection(P, G, 'I', weight=1.0)
    def su(P):
        return code()
    G._state_updater = su
    G._threshold = threshold_func
    
    M = MultiStateMonitor(G, record=True)
    Msp = SpikeMonitor(G)
    run(100*ms)
    print Msp.spikes
    M.plot()
    show()
