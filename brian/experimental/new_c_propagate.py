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

TODO:

* Have ConnectionCode object feature a C and Python code and namespace to
  be executed. The Python namespace can be used for grabbing some parameters
  such as the _cdi variables that currently have to be set in by hand.
'''

if __name__ == '__main__':
    from brian import *
    from brian.utils.documentation import *
    from brian.experimental.codegen.c_support_code import *
else:
    from ..connections import Connection, DelayConnection, MultiConnection, \
                DenseConnectionMatrix, DenseConstructionMatrix, \
                SparseConnectionMatrix, SparseConstructionMatrix, \
                DynamicConnectionMatrix, DynamicConstructionMatrix
    from ..log import log_debug, log_warn, log_info
    from ..utils.documentation import flattened_docstring
    from ..globalprefs import get_global_preference
    from codegen.c_support_code import *
import numpy
from scipy import weave
import new

__all__ = ['make_new_connection',
           'expand_code', 'transform_code',
           'iterate_over_spikes',
           'load_required_variables',
           'load_required_variables_delayedreaction',
           'load_required_variables_pastvalue',
           'iterate_over_row',
           'iterate_over_col',
           'ConnectionCode',
           ]


class ConnectionCode(object):
    def __init__(self, codestr, vars=None, pycodestr=None, pyvars=None):
        if pyvars is None:
            pyvars = {}
        if pycodestr is None:
            pycodestr = ''
        if vars is None:
            vars = {}
        self.vars = vars
        self.codestr = codestr
        if len([line for line in pycodestr.split('\n') if line]) > 1:
            pycodestr = flattened_docstring(pycodestr)
        else:
            pycodestr = pycodestr.strip()
        self.pycodestr = pycodestr
        self.pyvars = pyvars
        self.prepared = False
        self._weave_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._weave_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']

    def prepare(self):
        self.pyvars['vars'] = self.vars
        self.vars['_spikes'] = None
        self.vars['_spikes_len'] = None
        self.vars_list = self.vars.keys()
        if len(self.pycodestr):
            self.compiled_pycode = compile(self.pycodestr, 'ConnectionCode', 'exec')
        else:
            self.compiled_pycode = None
        self.prepared = True

    def __call__(self, _spikes):
        if not self.prepared:
            self.prepare()
        if len(_spikes):
            if not isinstance(_spikes, numpy.ndarray):
                _spikes = array(_spikes, dtype=int)
            vars = self.vars
#            print '****'
#            print self.codestr
#            for k, v in vars.iteritems():
#                if isinstance(v, numpy.ndarray):
#                    print k, ': shape =', v.shape
#                else:
#                    print k, ':', v
#            import sys
#            sys.stdout.flush()
            vars['_spikes'] = _spikes
            vars['_spikes_len'] = len(_spikes)
            if self.compiled_pycode is not None:
                exec self.compiled_pycode in self.pyvars
            weave.inline(self.codestr, self.vars_list,
                         local_dict=self.vars,
                         support_code=c_support_code,
                         compiler=self._weave_compiler,
                         extra_compile_args=self._extra_compile_args)

    def __str__(self):
        s = 'C code:\n'
        spaces = 0
        for line in self.codestr.split('\n'):
            if line.strip():
                if '}' in line: spaces -= 4
                s += ' ' * spaces + line.strip() + '\n'
                if '{' in line: spaces += 4
        s += 'Python code:\n'
        s += self.pycodestr
        return s
    __repr__ = __str__

def expand_code(code):
    if isinstance(code, ConnectionCode):
        return code
    elif isinstance(code, (tuple, list)):
        codestr = '\n'.join([expand_code(c).codestr for c in code])
        vars = {}
        pyvars = {}
        for c in code:
            vars.update(c.vars)
            pyvars.update(c.pyvars)
        pycodestr = '\n'.join([expand_code(c).pycodestr for c in code])
        return ConnectionCode(codestr, vars, pycodestr, pyvars)
    else:
        raise TypeError('ConnectionCode should be string or tuple')

def transform_code(codestr, vars=None):
    # TODO: replace with something more sophisticated than this
    return ConnectionCode('\n'.join(line + ';' for line in codestr.split('\n') if line.strip()), vars)

def iterate_over_spikes(neuron_index, spikes, code):
    outcode = '''
    for(int _spike_index=0; _spike_index<%SPIKES_LEN%; _spike_index++)
    {
        const int %NEURON_INDEX% = %SPIKES%[_spike_index];
        %CODE%
    }
    '''
    code = expand_code(code)
    outcode = outcode.replace('%SPIKES%', spikes)
    outcode = outcode.replace('%SPIKES_LEN%', spikes + '_len')
    outcode = outcode.replace('%NEURON_INDEX%', neuron_index)
    outcode = outcode.replace('%CODE%', code.codestr)
    return ConnectionCode(outcode, code.vars, code.pycodestr, code.pyvars)

def load_required_variables(neuron_index, neuron_vars):
    vars = {}
    codestr = ''
    for k, v in neuron_vars.iteritems():
        vars[k + '__array'] = v
        codestr += 'double &' + k + ' = ' + k + '__array[' + neuron_index + '];\n'
    return ConnectionCode(codestr, vars)

def load_required_variables_delayedreaction(neuron_index, delay, delay_index, neuron_var, C):
    vars = {}
    codestr = 'double &%Z% = _dr[((%CDI%+(int)(_idt*%D%))%_md)*%N%+%I%];'
    codestr = codestr.replace('%CDI%', delay_index)
    codestr = codestr.replace('%Z%', neuron_var)
    codestr = codestr.replace('%D%', delay)
    codestr = codestr.replace('%N%', '_num_target_neurons')
    codestr = codestr.replace('%I%', neuron_index)
    vars['_num_target_neurons'] = len(C.target)
    vars['_dr'] = C._delayedreaction
    vars[delay_index] = None # filled in by Python code (below)
    vars['_idt'] = C._invtargetdt
    vars['_md'] = C._max_delay
    pyvars = {'conn':C}
    pycodestr = "vars['_cdi'] = conn._cur_delay_ind"
    return ConnectionCode(codestr, vars, pycodestr, pyvars)

def load_required_variables_pastvalue(neuron_index, time, neuron_vars):
    vars = {}
    pyvars = {}
    pycodestr = ''
    codestr = ''
    for k, M in neuron_vars.iteritems():
        vars[k + '__values'] = M._values
        vars[k + '__arraylen'] = M._values.shape[1]
        vars[k + '__cti'] = None # current_time_index filled in by propagation function
        vars[k + '__idt'] = M._invtargetdt
        vars[k + '__nd'] = M.num_duration
        pyvars[k + '__RecentStateMonitor'] = M
        pycodestr += "vars['%k%__cti'] = %k%__RecentStateMonitor.current_time_index\n" .replace('%k%', k)
        newcodestr = 'double &%var% = %var%__values[((%var%__nd+%var%__cti-1-(int)(%var%__idt*%time%))%%var%__nd)*%var%__arraylen+%i%];\n'
        newcodestr = newcodestr.replace('%var%', k)
        newcodestr = newcodestr.replace('%time%', time)
        newcodestr = newcodestr.replace('%i%', neuron_index)
        codestr += newcodestr;
    return ConnectionCode(codestr, vars, pycodestr, pyvars)

def iterate_over_row(target_index, weight_variable, weight_matrix, source_index,
                     code, extravars={}):
    code = expand_code(code)
    vars = {}
    vars.update(code.vars)
    if isinstance(weight_matrix, DenseConnectionMatrix):
        outcode = '''
        double *_weight_arr_row = _weight_arr+%SOURCEINDEX%*_num_target_neurons;
        for(int %TARGETINDEX%=0; %TARGETINDEX%<_num_target_neurons; %TARGETINDEX%++)
        {
            double &%WEIGHT% = _weight_arr_row[%TARGETINDEX%];
            %EXTRAVARS%
            %CODE%
        }
        '''
        extravarscode = ''
        for k, v in extravars.iteritems():
            extracodetmp = 'double &%V% = %V%__array[%TARGETINDEX%+%SOURCEINDEX%*_num_target_neurons];'
            extracodetmp = extracodetmp.replace('%V%', k)
            extracodetmp = extracodetmp.replace('%TARGETINDEX%', target_index)
            extracodetmp = extracodetmp.replace('%SOURCEINDEX%', source_index)
            vars[k + '__array'] = numpy.asarray(v)
            extravarscode += extracodetmp
        outcode = outcode.replace('%EXTRAVARS%', extravarscode)
        vars['_weight_arr'] = numpy.asarray(weight_matrix)
        vars['_num_target_neurons'] = weight_matrix.shape[1]
    elif isinstance(weight_matrix, SparseConnectionMatrix):
        outcode = '''
        for(int _p=_rowind[%SOURCEINDEX%]; _p<_rowind[%SOURCEINDEX%+1]; _p++)
        {
            int %TARGETINDEX% = _allj[_p];
            double &%WEIGHT% = _alldata[_p];
            %EXTRAVARS%
            %CODE%
        }
        '''
        extravarscode = ''
        for k, v in extravars.iteritems():
            extracodetmp = 'double &%V% = %V%__alldata[_p];'
            extracodetmp = extracodetmp.replace('%V%', k)
            vars[k + '__alldata'] = v.alldata
            extravarscode += extracodetmp
        outcode = outcode.replace('%EXTRAVARS%', extravarscode)
        vars['_rowind'] = weight_matrix.rowind
        vars['_allj'] = weight_matrix.allj
        vars['_alldata'] = weight_matrix.alldata
    elif isinstance(weight_matrix, DynamicConnectionMatrix):
        # TODO: support dynamic matrix structure
        # the best way to support dynamic matrix type would be to
        # reorganise dynamic matrix data structure. Ideally, it should consist
        # of numpy arrays only. Maybe some sort of linked list structure?
        # Otherwise, we can use code that accesses the Python lists, but it's
        # probably less efficient (maybe this is anyway not a big issue with
        # the dynamic matrix type?)
        raise TypeError('Dynamic matrix not supported.')
    else:
        raise TypeError('Must be dense/sparse/dynamic matrix.')
    outcode = outcode.replace('%TARGETINDEX%', target_index)
    outcode = outcode.replace('%SOURCEINDEX%', source_index)
    outcode = outcode.replace('%WEIGHT%', weight_variable)
    outcode = outcode.replace('%CODE%', code.codestr)
    return ConnectionCode(outcode, vars, code.pycodestr, code.pyvars)

def iterate_over_col(source_index, weight_variable, weight_matrix, target_index,
                     code, extravars={}):
    code = expand_code(code)
    vars = {}
    vars.update(code.vars)
    if isinstance(weight_matrix, DenseConnectionMatrix):
        outcode = '''
        double *_weight_arr_row = _weight_arr+%TARGETINDEX%;
        for(int %SOURCEINDEX%=0; %SOURCEINDEX%<_num_source_neurons; %SOURCEINDEX%++)
        {
            double &%WEIGHT% = _weight_arr_row[%SOURCEINDEX%*_num_target_neurons];
            %EXTRAVARS%
            %CODE%
        }
        '''
        extravarscode = ''
        for k, v in extravars.iteritems():
            extracodetmp = 'double &%V% = %V%__array[%SOURCEINDEX%+%TARGETINDEX%*_num_target_neurons];'
            extracodetmp = extracodetmp.replace('%V%', k)
            extracodetmp = extracodetmp.replace('%SOURCEINDEX%', source_index)
            extracodetmp = extracodetmp.replace('%TARGETINDEX%', target_index)
            vars[k + '__array'] = numpy.asarray(v)
            extravarscode += extracodetmp
        outcode = outcode.replace('%EXTRAVARS%', extravarscode)
        vars['_weight_arr'] = numpy.asarray(weight_matrix)
        vars['_num_source_neurons'] = weight_matrix.shape[0]
        vars['_num_target_neurons'] = weight_matrix.shape[1]
    elif isinstance(weight_matrix, SparseConnectionMatrix):
        outcode = '''
        for(int _q=_colind[%TARGETINDEX%]; _q<_colind[%TARGETINDEX%+1]; _q++)
        {
            int _p = _allcoldataindices[_q];
            int %SOURCEINDEX% = _colalli[_q];
            double &%WEIGHT% = _alldata[_p];
            %EXTRAVARS%
            %CODE%
        }
        '''
        extravarscode = ''
        for k, v in extravars.iteritems():
            extracodetmp = 'double &%V% = %V%__alldata[_p];'
            extracodetmp = extracodetmp.replace('%V%', k)
            vars[k + '__alldata'] = v.alldata
            extravarscode += extracodetmp
        outcode = outcode.replace('%EXTRAVARS%', extravarscode)
        vars['_colind'] = weight_matrix.colind
        vars['_colalli'] = weight_matrix.colalli
        vars['_allcoldataindices'] = weight_matrix.allcoldataindices
        vars['_alldata'] = weight_matrix.alldata
    elif isinstance(weight_matrix, DynamicConnectionMatrix):
        # TODO: support dynamic matrix structure
        # the best way to support dynamic matrix type would be to
        # reorganise dynamic matrix data structure. Ideally, it should consist
        # of numpy arrays only. Maybe some sort of linked list structure?
        # Otherwise, we can use code that accesses the Python lists, but it's
        # probably less efficient (maybe this is anyway not a big issue with
        # the dynamic matrix type?)
        raise TypeError('Dynamic matrix not supported.')
    else:
        raise TypeError('Must be dense/sparse/dynamic matrix.')
    outcode = outcode.replace('%SOURCEINDEX%', source_index)
    outcode = outcode.replace('%TARGETINDEX%', target_index)
    outcode = outcode.replace('%WEIGHT%', weight_variable)
    outcode = outcode.replace('%CODE%', code.codestr)
    return ConnectionCode(outcode, vars, code.pycodestr, code.pyvars)

# TODO:
# * TEST iterate_over_col
# * TEST load_required_variables_pastvalue

def generate_connection_code(C):
    modulation = C._nstate_mod is not None
    delay = isinstance(C, DelayConnection)
    if not modulation and not delay:
        code = iterate_over_spikes('_j', '_spikes',
                     iterate_over_row('_k', 'w', C.W, '_j',
                         (load_required_variables('_k', {'V':C.target.state(C.nstate)}),
                          transform_code('V += w'))))
    elif modulation and not delay:
        code = iterate_over_spikes('_j', '_spikes',
                     (load_required_variables('_j', {'modulation':C.source.state(C._nstate_mod)}),
                      iterate_over_row('_k', 'w', C.W, '_j',
                         (load_required_variables('_k', {'V':C.target.state(C.nstate)}),
                          transform_code('V += w*modulation')))))
    elif not modulation and delay:
        code = iterate_over_spikes('_j', '_spikes',
                     iterate_over_row('_k', 'w', C.W, '_j', extravars={'_delay':C.delayvec},
                         code=(load_required_variables_delayedreaction('_k', '_delay', '_cdi', 'V', C),
                               transform_code('V += w'))))
    elif modulation and delay:
        code = iterate_over_spikes('_j', '_spikes',
                     (load_required_variables('_j', {'modulation':C.source.state(C._nstate_mod)}),
                      iterate_over_row('_k', 'w', C.W, '_j', extravars={'_delay':C.delayvec},
                         code=(load_required_variables_delayedreaction('_k', '_delay', '_cdi', 'V', C),
                               transform_code('V += w*modulation')))))
    else:
        raise TypeError('Not supported.')
    return code

def new_propagate(self, _spikes):
    if not self.iscompressed:
        self.compress()
    if not hasattr(self, '_connection_code'):
        self._connection_code = generate_connection_code(self)
        log_warn('brian.experimental.new_c_propagate', 'Using new C based propagation function.')
        log_debug('brian.experimental.new_c_propagate', 'C based propagation code:\n' + str(self._connection_code))
    if len(_spikes):
        self._connection_code(_spikes)

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

if __name__ == '__main__':

    log_level_debug()

    structure = 'sparse'
    delay = 0 * ms#0*ms or True
    modulation = False

    G = NeuronGroup(1, 'V:1\nmod:1', reset=0, threshold=1)
    G.V = 2
    G.mod = 5
    H = NeuronGroup(10, 'V:1')
    if modulation:
        modulation = 'mod'
    else:
        modulation = None
    C = Connection(G, H, 'V', structure=structure, delay=delay, modulation=modulation)
    C[0, :] = linspace(0, 1, 10)
    if delay:
        C.delay[0, :] = linspace(0, 1, 10) * ms

    M = StateMonitor(H, 'V', record=True)

    make_new_connection(C)

    run(2 * ms)

    M.plot()
    legend()
    show()
