from brian import *
from integration import *
from generation import *
from symbols import *

def make_integration_code_block(group, eqs, method, language):
    statements = make_integration_step(method, eqs)
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

class CodeGenStateUpdater(StateUpdater):
    def __init__(self, eqs, method, language, clock=None):
        self.clock = guess_clock(clock)
        self.eqs = eqs
        self.method = method
        self.language = language
        self.block = make_integration_code_block(group, eqs, method, language)
        self.prepared = False
    def __call__(self, G):
        if not self.prepared:
            self.symbols = get_neuron_group_symbols(G, self.language)
            self.code = self.block.generate(self.language, self.symbols)
            ns = self.code.namespace
            ns['dt'] = G.clock._dt
            ns['_num_neurons'] = len(G)
            self.prepared = True
        code = self.code
        ns = code.namespace
        ns['t'] = G.clock._t
        code()
