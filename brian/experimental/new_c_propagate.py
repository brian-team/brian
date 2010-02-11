''' 
NOTES:

General scheme for propagation code is something like:

Iterate over spikes (spike index i):
    Set source neuron index j=spikes[i]
    
    Load required source neuron variables for neuron index j
    
    Iterate over row j of W:
        Set target neuron index k
        Load weight variable
        
        Load required target neuron variables for neuron index k
        
        Execute propagation code

Functionally:

iterate_over_spikes('j', 'spikes',
    (load_required_variables('j', {'modulation':modulation_var}),
     iterate_over_row('k', 'w', W, 'j',
         (load_required_variables('k', {'V':V}),
          transform_code('V += w*modulation')
          )
         )
     )
    )

With some options like:

* load_required_variables('j', {})
* iterate_over_row('k', 'w', W, 'j', delayvec=delayvec, delayvar='delay', ...)
* load_required_variables_delayedreaction('k', {'V':{'dr':dr, 'cdi':cdi, 'idt':idt, 'md':md}}) 

We could also have:

* iterate_over_col('k', 'w', W, 'j', ...)

And STDP could be implemented as follows, for pre spikes:

iterate_over_spikes('j', 'spikes',
    (load_required_variables('j', {'A_pre':A_pre}),
     """
     A_pre += dA_pre
     """,
     iterate_over_row('k', 'w', W, 'j',
         (load_required_variables('k', {'A_post':A_post}),
          transform_code('w += A_post')
          )
         )
     )
    )

And for post spikes:

iterate_over_spikes('j', 'spikes',
    (load_required_variables('j', {'A_post':A_post}),
     transform_code('A_post += dA_post'),
     iterate_over_col('k', 'w', W, 'j',
         (load_required_variables('k', {'A_pre':A_pre}),
          transform_code('w += A_pre')
          )
         )
     )
    )

To do STDP with delays, we need also:

* load_required_variables_pastvalue('k', {'A_post':A_post_monitor})

Maybe for future-proofing STDP against having multiple per-synapse variables in
the future, and generally having per-synapse dynamics and linked matrices which
we want to jointly iterate over, could improve iterate_over_row/col to iterate
over several synaptic variables with the same underlying matrix structure as the
main weight matrix. Then, instead of having delayvec=delayvec, delayvar='delay'
as a special case, we'd have a list of additional linked matrices.
'''
if __name__=='__main__':
    from brian import *
else:
    from ..connection import Connection, DelayConnection, DenseConnectionMatrix, SparseConnectionMatrix, DenseConstructionMatrix, SparseConstructionMatrix, MultiConnection
    from ..log import log_warn, log_info
import numpy
from scipy import weave
import new

__all__ = ['make_new_connection']

class Code(object):
    def __init__(self, codestr, vars=None):
        if vars is None:
            vars = {}
        self.vars = vars
        self.codestr = codestr

def expand_code(code):
    if isinstance(code, Code):
        return code
    elif isinstance(code, (tuple, list)):
        codestr = '\n'.join([expand_code(c).codestr for c in code])
        vars = {}
        for c in code:
            vars.update(c.vars)
        return Code(codestr, vars)
    else:
        raise TypeError('Code should be string or tuple')

def transform_code(codestr):
    # TODO: replace with something more sophisticated than this
    return Code('\n'.join(line+';' for line in code.split('\n') if line.strip()))

def iterate_over_spikes(neuron_index, spikes, code):
    outcode = '''
    for(int _spike_index=0; _spike_index<%SPIKES_LEN%; _spike_index++)
    {
        %NEURON_INDEX% = %SPIKES%[_spike_index];
        %CODE%
    }
    '''
    code = expand_code(code)
    outcode = outcode.replace('%SPIKES%', spikes)
    outcode = outcode.replace('%SPIKES_LEN%', spikes+'_len')
    outcode = outcode.replace('%NEURON_INDEX%', neuron_index)
    outcode = outcode.replace('%CODE%', code.codestr)
    return Code(outcode, code.vars)

def load_required_variables(neuron_index, neuron_vars):
    vars = {}
    codestr = ''
    for k, v in neuron_vars.iteritems():
        vars[k+'__array'] = v
        codestr += 'double &'+k+' = '+k+'__array['+neuron_index+'];\n'
    return Code(codestr, vars)

# TODO:
# * iterate_over_row
# * iterate_over_col
# * load_required_variables_delayedreaction
# * load_required_variables_pastvalue

def generate_connection_code(C):
    modulation = C._nstate_mod is not None
    delay = isinstance(C, DelayConnection)
    vars = {
        '_spikes':None,
        '_nspikes':None,
        '_num_source_neurons':len(C.source),
        '_num_target_neurons':len(C.target),
        }
    if delay:
        vars['_dr'] = C._delayedreaction
        vars['_cdi'] = None # filled in by propagation function
        vars['_idt'] = C._invtargetdt
        vars['_md'] = C._max_delay
    else:
        vars['_target_array'] = C.target._S[C.nstate]
    code = '''
    for(int _spike_index=0; _spike_index<_nspikes; _spike_index++)
    {
        int _source_neuron_index = _spikes[_spike_index];
    '''
    if modulation:
        vars['_modulation_array'] = C.source._S[C._nstate_mod]
        code += '''
        double _modulation = _modulation_array[_source_neuron_index];
        '''
    if isinstance(C.W, DenseConnectionMatrix):
        vars['_weight_array'] = numpy.asarray(C.W)
        code += '''
        for(int _target_neuron_index=0; _target_neuron_index<_num_target_neurons; _target_neuron_index++)
        {
            double &_weight = _weight_array[_target_neuron_index+_source_neuron_index*_num_target_neurons];
        '''
        if delay:
            vars['_delay_array'] = numpy.asarray(C.delayvec)
            code += '''
            double _delay = _delay_array[_target_neuron_index+_source_neuron_index*_num_target_neurons];
            '''
    elif isinstance(C.W, SparseConnectionMatrix):
        vars['_rowind'] = C.W.rowind
        vars['_allj'] = C.W.allj
        vars['_alldata'] = C.W.alldata
        code += '''
        for(int _p=_rowind[_source_neuron_index]; _p<_rowind[_source_neuron_index+1]; _p++)
        {
            int _target_neuron_index = _allj[_p];
            double &_weight = _alldata[_p];
        '''
        if delay:
            vars['_alldata_delay'] = C.delayvec.alldata
            code += '''
            double _delay = _alldata_delay[_p];
            '''
    else:
        raise TypeError('Not supported.')
    if delay:
        code += '''
            double &_target_var = _dr[((_cdi+(int)(_idt*_delay))%_md)*_num_target_neurons+_target_neuron_index];
        '''
    else:
        code += '''
            double &_target_var = _target_array[_target_neuron_index];
        '''
    if modulation:
        code += '''
            _target_var += _weight*_modulation;
        '''
    else:
        code += '''
            _target_var += _weight;
        '''
    code += '''
        }
    '''
    code += '''
    }
    '''
    code = '\n'.join(line for line in code.split('\n') if line.strip())
    return vars, code

def new_propagate(self, _spikes):
    if not self.iscompressed:
        self.compress()
    if not hasattr(self, '_vars'):
        self._vars, self._code = generate_connection_code(self)
        self._vars_list = self._vars.keys()
        log_warn('brian.experimental.new_c_propagate', 'Using new C based propagation function.')
        log_info('brian.experimental.new_c_propagate', 'C based propagation function code:\n'+self._code)
    if len(_spikes):
        if not isinstance(_spikes, numpy.ndarray):
            _spikes = array(_spikes, dtype=int)
        self._vars['_spikes'] = _spikes
        self._vars['_nspikes'] = len(_spikes)
        if isinstance(self, DelayConnection):
            self._vars['_cdi'] = self._cur_delay_ind
        weave.inline(self._code, self._vars_list,
                     local_dict=self._vars,
                     compiler='gcc',
                     extra_compile_args=['-O3'])

def make_new_connection(C):
    if C.__class__ is MultiConnection:
        for c in C.connections:
            make_new_connection(c)
    if C.__class__ is Connection or C.__class__ is DelayConnection:
        if C.W.__class__ is SparseConnectionMatrix or \
           C.W.__class__ is DenseConnectionMatrix or \
           C.W.__class__ is SparseConstructionMatrix or \
           C.W.__class__ is DenseConstructionMatrix:
            C.propagate = new.instancemethod(new_propagate, C, C.__class__)

if __name__=='__main__':
    
    structure = 'sparse'
    delay = False
    
    G = NeuronGroup(1, 'V:1', reset=0, threshold=1)
    G.V = 2
    H = NeuronGroup(10, 'V:1')
    C = Connection(G, H, 'V', structure=structure, delay=delay)
    C[0, :] = linspace(0, 1, 10)
    if delay:
        C.delay[0, :] = linspace(0, 1, 10)*ms
        
    M = StateMonitor(H, 'V', record=True)
    
    make_new_connection(C)
    
    run(2*ms)
    
    M.plot()
    legend()
    show()
