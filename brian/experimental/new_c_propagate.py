if __name__=='__main__':
    from brian import *
else:
    from ..connection import Connection, DelayConnection, DenseConnectionMatrix, SparseConnectionMatrix, DenseConstructionMatrix, SparseConstructionMatrix, MultiConnection
    from ..log import log_warn, log_info
import numpy
from scipy import weave
import new

__all__ = ['make_new_connection']

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
