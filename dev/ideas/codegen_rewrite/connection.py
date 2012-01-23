'''

SIMPLEST WAY:

    - Use something like previous codegen propagation stuff to generate
      basic template, which will resolve _synapse_index and _target_index
      (and also _source_index).
    - Use current code generation stuff once that has been generated.
    
BETTER WAY:

    - Allow Symbols to introduce loops in resolving dependencies, and make
      _synapse_index and _target_index into Symbols.
    - Requires a change of syntax to Symbol, may make it more complicated
    - On the other hand, the new propagation code is ugly, and needs to be
      adapted to work for Python as well

With better way, transformations would be something like (for C)

1.  Starting point
        V += w;
    Deps: V, w

2.  Resolve V
        double &V = _arr_V[_target_index];
        V += w;
    Deps: w, _target_index (_arr_V added to namespace)

3.  Resolve w
        double &w = _alldata_w[_synapse_index];
        double &V = _arr_V[_target_index];
        V += w;
    Deps: _target_index, _synapse_index (_alldata_w added to namespace)

4.  Resolve _target_index
        int target_index = _allj_w[_synapse_index];
        double &w = _alldata_w[_synapse_index];
        double &V = _arr_V[_target_index];
        V += w;
    Deps: _synapse_index (_allj_w added to namespace)
    
5.  Resolve _synapse_index
        for(int _synapse_index=_rowind_w[_source_index];
                _synapse_index<_rowind_w[_source_index+1];
                _synapse_index++)
        {
            int target_index = w_allj[_synapse_index];
            double &w = _alldata_w[_synapse_index];
            double &V = _arr_V[_target_index];
            V += w;
        }
    Deps: _source_index (_rowind_w added to namespace)
    
6.  Resolve _source_index
        int _source_index = _spikes[_spike_index];
        for(int _synapse_index=_rowind_w[_source_index];
                _synapse_index<_rowind_w[_source_index+1];
                _synapse_index++)
        {
            int target_index = w_allj[_synapse_index];
            double &w = _alldata_w[_synapse_index];
            double &V = _arr_V[_target_index];
            V += w;
        }
    Deps: _spike_index (_spikes added to namespace)
    
7.  Resolve _spike_index
        for(int _spike_index=0; _spike_index<_spikes_len; _spike_index++)
        {
            int _source_index = _spikes[_spike_index];
            for(int _synapse_index=_rowind_w[_source_index];
                    _synapse_index<_rowind_w[_source_index+1];
                    _synapse_index++)
            {
                int target_index = w_allj[_synapse_index];
                double &w = _alldata_w[_synapse_index];
                double &V = _arr_V[_target_index];
                V += w;
            }
        }
    Deps: none (_spikes_len added to namespace)

Same thing in Python:

1.  Starting point
        V += w;
    Deps: V, w
    
2.  Resolve V
        V[_target_index] += w
    Deps: w, _target_index
    
3.  Resolve w
        w = _alldata_w[_synapse_index]
        V[_target_index] += w
    Deps: _target_index, _synapse_index (_alldata_w added to namespace)

4.  Resolve _target_index
        _target_index = _allj_w[_synapse_index]
        w = _alldata_w[_synapse_index]
        V[_target_index] += w
    Deps: _synapse_index (_allj_w added to namespace)
    
5.  Resolve _synapse_index
        _synapse_index = slice(_rowind_w[_source_index], _rowind_w[_source_index+1])
        _target_index = _allj_w[_synapse_index]
        w = _alldata_w[_synapse_index]
        V[_target_index] += w
    Deps: _source_idnex (_rowind_w added to namespace)

6.  Resolve _source_index
        for _source_index in _spikes:
            _synapse_index = slice(_rowind_w[_source_index], _rowind_w[_source_index+1])
            _target_index = _allj_w[_synapse_index]
            w = _alldata_w[_synapse_index]
            V[_target_index] += w
    Deps: none (_spikes added to namespace)

Dependencies:
    _synapse_index -> _rowind_w, _source_index
    _target_index -> _allj_w, _synapse_index
    w -> _synapse_index
    V -> _target_index

Python:
    for _source_index in _spikes:
        _synapse_index = slice(_rowind_w[_source_index], _rowind_w[_source_index+1])
        _target_index = _allj_w[_synapse_index]
        w = _alldata_w[_synapse_index]
        V[_target_index] += w

C:
    for(int _spike_index=0; _spike_index<_spikes_len; _spike_index++)
    {
        int _source_index = _spikes[_spike_index];
        for(int _row_index=_rowind_w[_source_index];
                _row_index<_rowind_w[_source_index+1];
                _row_index++)
        {
            int target_index = w_allj[_row_index];
            double &V = _arr_V[_target_index];
            double &w = _alldata_w[_synapse_index];
            V += w;
        }
    }


'''
from brian import *
from symbols import *

class SparseConnectionMatrixValueSymbol(Symbol):
    def __init__(self, matrix, name, language):
        self.matrix = matrix
        Symbol.__init__(self, name, language)
    def update_namespace(self, namespace):
        namespace['_alldata_'+self.name] = self.matrix.alldata
    @property
    def load(self):
        if self.language.name=='python':
            template = '{name} = _alldata_{name}[_synapse_index]'
        elif self.language.name=='c':
            template = 'double &{name} = _alldata_{name}[_synapse_index];'
        return template.format(name=self.name)
    @property
    def write(self):
        if self.language.name=='python':
            template = '{name} = _alldata_{name}[_synapse_index]'
            return template.format(name=self.name)
        elif self.language.name=='c':
            return self.name
    @property
    def depends(self):
        return ['_synapse_index']

class CodeGenConnection(Connection):
    def propagate(self, spikes):
        Connection.propagate(self, spikes)

def make_connection_code_block(group, eqs, reset, language):
    pass
