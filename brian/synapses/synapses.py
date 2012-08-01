'''
The Synapses class - see BEP-21
'''
import re
import warnings
from operator import isSequenceType

import numpy as np
from numpy.random import binomial
from random import sample
from scipy import rand, randn

from brian.inspection import get_identifiers, namespace
from brian.log import log_debug
from brian.neurongroup import NeuronGroup
from brian.optimiser import AffineFunction, symbolic_eval
from brian.stdunits import ms
from brian.synapses.spikequeue import SpikeQueue
from brian.synapses.synaptic_equations import SynapticEquations
from brian.synapses.synapticvariable import (SynapticDelayVariable, 
                                             SynapticVariable, slice_to_array)
from brian.utils.documentation import flattened_docstring
from brian.utils.dynamicarray import DynamicArray, DynamicArray1D 

try:
    import sympy
    use_sympy = True
except:
    warnings.warn('sympy not installed: some features in Synapses will not be available')
    use_sympy = False

__all__ = ['Synapses','invert_array']

class Synapses(NeuronGroup): # This way we inherit a lot of useful stuff
    '''Set of synapses between two neuron groups
    
    Initialised with arguments:
    
    ``source``
        The source :class:`NeuronGroup`.
    ``target=None``
        The target :class:`NeuronGroup`. By default, target=source.
    ``model=None``
        The equations that defined the synaptic variables, as an Equations object or a string.
        The syntax is the same as for a :class:`NeuronGroup`.
    ``pre=None``
        The code executed when presynaptic spikes arrive at the synapses.
        There can be multiple presynaptic codes, passed as a list or tuple of strings.
    ``post=None``
        The code executed when postsynaptic spikes arrive at the synapses.
    ``max_delay=0*ms``
        The maximum pre and postsynaptic delay. This is only useful if the delays can change
        during the simulation.
    ``level=0``
        See :class:`Equations` for details.
    ``clock=None``
        The clock for updating synaptic state variables according to ``model``.
        Currently, this must be identical to both the source and target clocks.
    ``compile=False``
        Whether or not to attempt to compile the differential equation
        solvers (into Python code). Typically, for best performance, both ``compile``
        and ``freeze`` should be set to ``True`` for nonlinear differential equations.
    ``freeze=False``
        If True, parameters are replaced by their values at the time
        of initialization.
    ``method=None``
        If not None, the integration method is forced. Possible values are
        linear, nonlinear, Euler, exponential_Euler (overrides implicit and order
        keywords).
    ``unit_checking=True``
        Set to ``False`` to bypass unit-checking.
    ``order=1``
        The order to use for nonlinear differential equation solvers.
        TODO: more details.
    ``implicit=False``
        Whether to use an implicit method for solving the differential
        equations. TODO: more details.
    ``code_namespace=None``
        Namespace for the pre and post codes.
        
    **Methods**
    
    .. method:: state(var)

        Returns the vector of values for state
        variable ``var``, with length the number of synapses. The
        vector is an instance of class :class:`SynapticVariable`.
        
    .. method:: synapse_index(i)

        Returns the synapse indexes correspond to i, which can be a tuple or a slice.
        If i is a tuple (m,n), m and n can be an integer, an array, a slice or a subgroup.
    
    The following usages are also possible for a Synapses object ``S``:
    
    ``len(S)``
        Returns the number of synapses in ``S``.
        
    Attributes:
    
    ``delay``
        The presynaptic delays for all synapses (synapse->delay). If there are multiple
        presynaptic delays (multiple pre codes), this is a list.
    ``delay_pre``
        Same as ``delay``.
    ``delay_post``
        The postsynaptic delays for all synapses (synapse->delay post).
    ``lastupdate``
        The time of last update of all synapses (synapse->last update). This
        only exists if there are dynamic synaptic variables.
    
    Internal attributes:
    
    ``source``
        The source neuron group.
    ``target``
        The target neuron group.
    ``_S``
        The state matrix (a 2D dynamical array with values of synaptic variables).
        At run time, it is transformed into a static 2D array (with compress()).
    ``presynaptic``
        The (dynamic) array of presynaptic neuron indexes for all synapses (synapse->i).
    ``postsynaptic``
        The array of postsynaptic neuron indexes for all synapses (synapse->j).
    ``synapses_pre``
        A list of (dynamic) arrays giving the set of synapse indexes for each presynaptic neuron i
        (i->synapses)
    ``synapses_post``
        A list of (dynamic) arrays giving the set of synapse indexes for each postsynaptic neuron j
        (j->synapses)
    ``queues``
        List of SpikeQueues for pre and postsynaptic spikes.
    ``codes``
        The compiled codes to be executed on pre and postsynaptic spikes.
    ``namespaces``
        The namespaces for the pre and postsynaptic codes.
    '''
    def __init__(self, source, target = None, model = None, pre = None, post = None,
             max_delay = 0*ms,
             level = 0,
             clock = None,code_namespace=None,
             unit_checking = True, method = None, freeze = False, implicit = False, order = 1): # model (state updater) related
        
        target=target or source # default is target=source

        # Check clocks. For the moment we enforce the same clocks for all objects
        clock = clock or source.clock
        if source.clock!=target.clock:
            raise ValueError,"Source and target groups must have the same clock"

        if pre is None:
            pre_list=[]
        elif isSequenceType(pre) and not isinstance(pre,str): # a list of pre codes
            pre_list=pre
        else:
            pre_list=[pre]

        pre_list=[flattened_docstring(pre) for pre in pre_list]
        if post is not None:
            post=flattened_docstring(post)

        # Pre and postsynaptic indexes (synapse -> pre/post)
        self.presynaptic=DynamicArray1D(0,dtype=smallest_inttype(len(source))) # this should depend on number of neurons
        self.postsynaptic=DynamicArray1D(0,dtype=smallest_inttype(len(target))) # this should depend on number of neurons

        if not isinstance(model,SynapticEquations):
            model=SynapticEquations(model,level=level+1)
        # Insert the lastupdate variable if necessary (if it is mentioned in pre/post, or if there is event-driven code)
        expr=re.compile(r'\blastupdate\b')
        if (len(model._eventdriven)>0) or \
           any([expr.search(pre) for pre in pre_list]) or \
           (post is not None and expr.search(post) is not None):
            model+='\nlastupdate : second\n'
            pre_list=[pre+'\nlastupdate=t\n' for pre in pre_list]
            if post is not None:
                post=post+'\nlastupdate=t\n'
        
        # Identify pre and post variables in the model string
        # They are identified by _pre and _post suffixes
        # or no suffix for postsynaptic variables
        ids=[]
        for RHS in model._string.itervalues():
            ids.extend(get_identifiers(RHS))
        pre_ids=[id[:-4] for id in ids if id[-4:]=='_pre']
        post_ids=[id[:-5] for id in ids if id[-5:]=='_post']
        post_vars=[var for var in source.var_index if isinstance(var,str)] # postsynaptic variables
        post_ids2=list(set(ids).intersection(set(post_vars))) # post variables without the _post suffix

        # Insert static equations for pre and post variables
        S=self
        for name in pre_ids:
            model.add_eq(name+'_pre', 'S.source.'+name+'[S.presynaptic[:]]', source.unit(name),
                         global_namespace={'S':S})
        for name in post_ids:
            model.add_eq(name+'_post', 'S.target.'+name+'[S.postsynaptic[:]]', target.unit(name),
                         global_namespace={'S':S})
        for name in post_ids2: # we have to change the name of the variable to avoid problems with equation processing
            if name not in model._string: # check that it is not already defined
                model.add_eq(name, 'S.target.state_(__'+name+')[S.postsynaptic[:]]', target.unit(name),
                             global_namespace={'S':S,'__'+name:name})
        
        self.source=source
        self.target=target
        
        NeuronGroup.__init__(self, 0,model=model,clock=clock,level=level+1,unit_checking=unit_checking,method=method,freeze=freeze,implicit=implicit,order=order)
        '''
        At this point we have:
        * a state matrix _S with all variables
        * units, state dictionary with each value being a row of _S + the static equations
        * subgroups of synapses
        * link_var (i.e. we can link two synapses objects)
        * __len__
        * __setattr__: we can write S.w=array of values
        * var_index is a dictionary from names to row index in _S
        * num_states()
        
        Things we have that we don't want:
        * LS structure (but it will not be filled since the object does not spike)
        * (from Group) __getattr_ needs to be rewritten
        * a complete state updater, but we need to extract parameters and event-driven parts
        * The state matrix is not dynamic
        
        Things we may need to add:
        * _pre and _post suffixes
        '''       
        self._iscompressed=False # True if compress() has already been called
        
        # Look for event-driven code in the differential equations
        if use_sympy:
            eqs=self._eqs # an Equations object
            #vars=eqs._diffeq_names_nonzero # Dynamic variables
            vars=eqs._eventdriven.keys()
            var_set=set(vars)
            for var,RHS in eqs._eventdriven.iteritems():
                ids=get_identifiers(RHS)
                if len(set(list(ids)+[var]).intersection(var_set))==1:
                    # no external dynamic variable
                    # Now we test if it is a linear equation
                    _namespace=dict.fromkeys(ids,1.) # there is a possibility of problems here (division by zero)
                    # plus units problems? (maybe not since these are identifiers too)
                    # another option is to use random numbers, but that doesn't solve all problems
                    _namespace[var]=AffineFunction()
                    try:
                        eval(RHS,eqs._namespace[var],_namespace)
                    except: # not linear
                        raise TypeError,"Cannot turn equation for "+var+" into event-driven code"
                    z=symbolic_eval(RHS)
                    symbol_var=sympy.Symbol(var)
                    symbol_t=sympy.Symbol('t')-sympy.Symbol('lastupdate')
                    b=z.subs(symbol_var,0)
                    a=sympy.simplify(z.subs(symbol_var,1)-b)
                    if a==0:
                        expr=symbol_var+b*symbol_t
                    else:
                        expr=-b/a+sympy.exp(a*symbol_t)*(symbol_var+b/a)
                    expr=var+'='+str(expr)
                    # Replace pre and post code
                    # N.B.: the differential equations are kept, we will probably want to remove them!
                    pre_list=[expr+'\n'+pre for pre in pre_list]
                    if post is not None:
                        post=expr+'\n'+post
                else:
                    raise TypeError,"Cannot turn equation for "+var+" into event-driven code"
        elif len(self._eqs._eventdriven)>0:
            raise TypeError,"The Sympy package must be installed to produce event-driven code"

        if len(self._eqs._diffeq_names_nonzero)==0:
            self._state_updater=None
        
        # Set last spike to -infinity
        if 'lastupdate' in self.var_index:
            self.lastupdate=-1e6
        # _S is turned to a dynamic array - OK this is probably not good! we may lose references at this point
        S=self._S
        self._S=DynamicArray(S.shape)
        self._S[:]=S

        # Pre and postsynaptic delays (synapse -> delay_pre/delay_post)
        self._delay_pre=[DynamicArray1D(len(self),dtype=np.int16) for _ in pre_list] # max 32767 delays
        self._delay_post=DynamicArray1D(len(self),dtype=np.int16) # Actually only useful if there is a post code!
        
        # Pre and postsynaptic synapses (i->synapse indexes)
        max_synapses=2147483647 # it could be explicitly reduced by a keyword
        # We use a loop instead of *, otherwise only 1 dynamic array is created
        self.synapses_pre=[DynamicArray1D(0,dtype=smallest_inttype(max_synapses)) for _ in range(len(self.source))]
        self.synapses_post=[DynamicArray1D(0,dtype=smallest_inttype(max_synapses)) for _ in range(len(self.target))]

        # Code generation
        self._binomial = lambda n,p:np.random.binomial(np.array(n,dtype=int),p)

        self.contained_objects = []
        self.codes=[]
        self.namespaces=[]
        self.queues=[]
        for i,pre in enumerate(pre_list):
            code,_namespace=self.generate_code(pre,level+1,code_namespace=code_namespace)
            self.codes.append(code)
            self.namespaces.append(_namespace)
            self.queues.append(SpikeQueue(self.source, self.synapses_pre, self._delay_pre[i], max_delay = max_delay))
        
        if post is not None:
            code,_namespace=self.generate_code(post,level+1,direct=True,code_namespace=code_namespace)
            self.codes.append(code)
            self.namespaces.append(_namespace)
            self.queues.append(SpikeQueue(self.target, self.synapses_post, self._delay_post, max_delay = max_delay))

        self.queues_namespaces_codes = zip(self.queues,
                                           self.namespaces,
                                           self.codes)

        self.contained_objects+=self.queues
      
    def generate_code(self,code,level,direct=False,code_namespace=None):
        '''
        Generates pre and post code.
        
        ``code''
            The code as a string.
            
        ``level''
            The namespace level in which the code is executed.
        
        ``direct=False''
            If True, the code is generated assuming that
            postsynaptic variables are not modified. This makes the
            code faster.
            
        ``code_namespace''
            Additional namespace (highest priority)
        
        TODO:
        * include static variables (substitution)
        * have a list of variable names
        '''
        # Handle multi-line pre, post equations and multi-statement equations separated by ;
        # (this should probably be factored)
        if '\n' in code:
            code = flattened_docstring(code)
        elif ';' in code:
            code = '\n'.join([line.strip() for line in code.split(';')])
        
        # Create namespaces
        _namespace = namespace(code, level = level + 1)
        if code_namespace is not None:
            _namespace.update(code_namespace)
        _namespace['target'] = self.target # maybe we could save one indirection here
        _namespace['unique'] = np.unique
        _namespace['nonzero'] = np.nonzero
        _namespace['empty'] = np.empty
        _namespace['logical_not'] = np.logical_not
        _namespace['not_equal'] = np.not_equal
        _namespace['take'] = np.take
        _namespace['extract'] = np.extract
        _namespace['add'] = np.add
        _namespace['hstack'] = np.hstack

        code = re.sub(r'\b' + 'rand\(\)', 'rand(n)', code)
        code = re.sub(r'\b' + 'randn\(\)', 'randn(n)', code)

        # Generate the code
        def update_code(code, indices, postinds):
            res = code
            # given the synapse indices, write the update code,
            # this is here because in the code we generate we need to write this twice (because of the multiple presyn spikes for the same postsyn neuron problem)
                       
            # Replace synaptic variables by their value
            for var in self.var_index: # static variables are not included here
                if isinstance(var, str):
                    res = re.sub(r'\b' + var + r'\b', var + '['+indices+']', res) # synaptic variable, indexed by the synapse number
 
            # Replace postsynaptic variables by their value
            for postsyn_var in self.target.var_index: # static variables are not included here
                if isinstance(postsyn_var, str):
                    #res = re.sub(r'\b' + postsyn_var + r'_post\b', 'target.' + postsyn_var + '['+postinds+']', res)# postsyn variable, indexed by post syn neuron numbers
                    #res = re.sub(r'\b' + postsyn_var + r'\b', 'target.' + postsyn_var + '['+postinds+']', res)# postsyn variable, indexed by post syn neuron numbers
                    res = re.sub(r'\b' + postsyn_var + r'_post\b', '_target_' + postsyn_var + '['+postinds+']', res)# postsyn variable, indexed by post syn neuron numbers
                    res = re.sub(r'\b' + postsyn_var + r'\b', '_target_' + postsyn_var + '['+postinds+']', res)# postsyn variable, indexed by post syn neuron numbers
                    _namespace['_target_' + postsyn_var] = self.target.state_(postsyn_var)
            
            # Replace presynaptic variables by their value
            for presyn_var in self.source.var_index: # static variables are not included here
                if isinstance(presyn_var, str):
                    #res = re.sub(r'\b' + presyn_var + r'_pre\b', 'source.' + presyn_var + '[_pre['+indices+']]', res)# postsyn variable, indexed by post syn neuron numbers
                    res = re.sub(r'\b' + presyn_var + r'_pre\b', '_source_' + presyn_var + '[_pre['+indices+']]', res)# postsyn variable, indexed by post syn neuron numbers
                    _namespace['_source_' + presyn_var] = self.source.state_(presyn_var)
 
            # Replace n by number of synapses being updated
            res = re.sub(r'\bn\b','len('+indices+')', res)
 
            return res
 
        if direct: # direct update code, not caring about multiple accesses to postsynaptic variables
            code_str = '_post_neurons = _post[_synapses]\n'+update_code(code, '_synapses', '_post_neurons') + "\n"            
        else:
            algo = 3
            if algo==0:
                ## Old version using numpy's unique()
                code_str = "_post_neurons = _post[_synapses]\n" # not necessary to do a copy because _synapses is not a slice
                code_str += "_u, _i = unique(_post_neurons, return_index = True)\n"
                #code_str += update_code(code, '_synapses[_i]', '_u') + "\n"
                code_str += update_code(code, '_synapses[_i]', '_post[_synapses[_i]]') + "\n"
                code_str += "if len(_u) < len(_post_neurons):\n"
                code_str += "    _post_neurons[_i] = -1\n"
                code_str += "    while (len(_u) < len(_post_neurons)) & (_post_neurons>-1).any():\n" # !! the any() is time consuming (len(u)>=1??)
                #code_str += "    while (len(_u) < len(_post_neurons)) & (len(_u)>1):\n" # !! the any() is time consuming (len(u)>=1??)
                code_str += "        _u, _i = unique(_post_neurons, return_index = True)\n"
                code_str += indent(update_code(code, '_synapses[_i[1:]]', '_post[_synapses[_i[1:]]]'),2) + "\n"
                code_str += "        _post_neurons[_i[1:]] = -1 \n"
            elif algo==1:
                code_str = "_post_neurons = _post[_synapses]\n" # not necessary to do a copy because _synapses is not a slice
                code_str += "_perm = _post_neurons.argsort()\n"
                code_str += "_aux = _post_neurons[_perm]\n"
                code_str += "_flag = empty(len(_aux) + 1, dtype = bool)\n"
                code_str += "_flag[0] = _flag[-1] = True\n"
                code_str += "not_equal(_aux[1:], _aux[:-1], _flag[1:-1])\n"
                code_str += "_F = _flag.nonzero()[0][:-1]\n"
                code_str += "logical_not(_flag, _flag)\n"
                code_str += "while len(_F):\n"
                code_str += "    _u = _aux[_F]\n"
                code_str += "    _i = _perm[_F]\n"
                code_str += indent(update_code(code, '_synapses[_i]', '_u'), 1) + "\n"
                code_str += "    _F += 1\n"
                code_str += "    _F = _F[_flag[_F]]\n"
            elif algo==2:
                code_str = '''
                _post_neurons = _post.data.take(_synapses)
                _perm = _post_neurons.argsort()
                _aux = _post_neurons.take(_perm)
                _flag = empty(len(_aux)+1, dtype=bool)
                _flag[0] = _flag[-1] = 1
                not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
                if 0:#_flag.sum()==len(_aux)+1:
                %(code1)s
                else:
                    _F = _flag.nonzero()[0][:-1]
                    logical_not(_flag, _flag)
                    while len(_F):
                        _u = _aux.take(_F)
                        _i = _perm.take(_F)
                %(code2)s
                        _F += 1
                        _F = extract(_flag.take(_F), _F)
                '''
                code_str = flattened_docstring(code_str) % {'code1': indent(update_code(code, '_synapses','_post_neurons'), 1),
                                                            'code2': indent(update_code(code, '_synapses[_i]', '_u'), 2)}
            elif algo==3:
                code_str = '''
                _post_neurons = _post.data.take(_synapses)
                _perm = _post_neurons.argsort()
                _aux = _post_neurons.take(_perm)
                _flag = empty(len(_aux)+1, dtype=bool)
                _flag[0] = _flag[-1] = 1
                not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
                _F = _flag.nonzero()[0][:-1]
                logical_not(_flag, _flag)
                while len(_F):
                    _u = _aux.take(_F)
                    _i = _perm.take(_F)
                %(code)s
                    _F += 1
                    _F = extract(_flag.take(_F), _F)
                '''
                code_str = flattened_docstring(code_str) % {'code': indent(update_code(code, '_synapses[_i]', '_u'), 1)}
            elif algo==4:
                code_str = '''
                _post_neurons = _post[_synapses]
                _perm = _post_neurons.argsort()
                _aux = _post_neurons[_perm]
                _flag = empty(len(_aux)+1, dtype=bool)
                _flag[0] = _flag[-1] = 1
                not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
                _F = _flag.nonzero()[0][:-1]
                logical_not(_flag, _flag)
                while len(_F):
                    _u = _aux[_F]
                    _i = _perm[_F]
                %(code)s
                    _F += 1
                    _F = _F[_flag[_F]]
                '''
                code_str = flattened_docstring(code_str) % {'code': indent(update_code(code, '_synapses[_i]', '_u'), 1)}
#        print code_str
            
        log_debug('brian.synapses', '\nCODE:\n'+code_str)
        
        # Compile
        compiled_code = compile(code_str, "Synaptic code", "exec")
        
        return compiled_code,_namespace

    def __setitem__(self, key, value):
        '''
        Creates new synapses.
        Synapse indexes are created such that synapses with the same presynaptic neuron
        and delay have contiguous indexes.
        
        Caution:
        1) there is no deletion
        2) synapses are added, not replaced (e.g. S[1,2]=True;S[1,2]=True creates 2 synapses)
        
        TODO:
        * S[:,:]=array (boolean or int)
        '''
        if self._iscompressed:
            raise AttributeError,"Synapses cannot be added after they have been run"
        
        if not isinstance(key, tuple): # we should check that number of elements is 2 as well
            raise AttributeError,'Synapses behave as 2-D objects'
        pre,post=key # pre and post indexes (can be slices)
        
        '''
        Each of these sets of statements creates:
        * synapses_pre: a mapping from presynaptic neuron to synapse indexes
        * synapses_post: same
        * presynaptic: an array of presynaptic neuron indexes (synapse->pre)
        * postsynaptic: same
        '''
        pre_slice = self.presynaptic_indexes(pre)
        post_slice = self.postsynaptic_indexes(post)
        # Bound checks
        if pre_slice[-1]>=len(self.source):
            raise ValueError('Presynaptic index %d greater than number of '\
                             'presynaptic neurons (%d)'
                             % (pre_slice[-1], len(self.source)))
        if post_slice[-1]>=len(self.target):
            raise ValueError('Postsynaptic index %d greater than number of '\
                             'postsynaptic neurons (%d)'
                             % (post_slice[-1], len(self.target)))

        if isinstance(value,float):
            self.connect_random(pre,post,value)
            return
        elif isinstance(value, (int, bool)): # ex. S[1,7]=True
            # Simple case, either one or multiple synapses between different neurons
            if value is False:
                raise ValueError('Synapses cannot be deleted')
            elif value is True:
                nsynapses = 1
            else:
                nsynapses = value

            postsynaptic,presynaptic=np.meshgrid(post_slice,pre_slice) # synapse -> pre, synapse -> post
            # Flatten
            presynaptic.shape=(presynaptic.size,)
            postsynaptic.shape=(postsynaptic.size,)
            # pre,post -> synapse index, relative to last synapse
            # (that's a complex vectorised one!)
            synapses_pre=np.arange(len(presynaptic)).reshape((len(pre_slice),len(post_slice)))
            synapses_post=np.ones((len(post_slice),1),dtype=int)*np.arange(0,len(presynaptic),len(post_slice))+\
                          np.arange(len(post_slice)).reshape((len(post_slice),1))
            # Repeat
            if nsynapses>1:
                synapses_pre=np.hstack([synapses_pre+k*len(presynaptic) for k in range(nsynapses)]) # could be vectorised
                synapses_post=np.hstack([synapses_post+k*len(presynaptic) for k in range(nsynapses)]) # could be vectorised
                presynaptic=np.tile(presynaptic,nsynapses)
                postsynaptic=np.tile(postsynaptic,nsynapses)
            # Make sure the type is correct
            synapses_pre=np.array(synapses_pre,dtype=self.synapses_pre[0].dtype)
            synapses_post=np.array(synapses_post,dtype=self.synapses_post[0].dtype)
            # Turn into dictionaries
            synapses_pre=dict(zip(pre_slice,synapses_pre))
            synapses_post=dict(zip(post_slice,synapses_post))
        elif isinstance(value,str): # string code assignment
            # For subgroups, origin of i and j are shifted to subgroup origin
            if isinstance(pre,NeuronGroup):
                pre_shift=pre_slice[0]
            else:
                pre_shift=0
            if isinstance(post,NeuronGroup):
                post_shift=post_slice[0]
            else:
                post_shift=0
            code = re.sub(r'\b' + 'rand\(\)', 'rand(n)', value) # replacing rand()
            code = re.sub(r'\b' + 'randn\(\)', 'randn(n)', code) # replacing randn()
            _namespace = namespace(value, level=1)
            _namespace.update({'j' : post_slice-post_shift,
                               'n' : len(post_slice),
                               'rand': np.random.rand,
                               'randn': np.random.randn})
#            try: # Vectorise over all indexes: not faster! 
#                post,pre=np.meshgrid(post_slice-post_shift,pre_slice-pre_shift)
#                pre=pre.flatten()
#                post=post.flatten()
#                _namespace['i']=array(pre,dtype=self.presynaptic.dtype)
#                _namespace['j']=array(post,dtype=self.postsynaptic.dtype)
#                _namespace['n']=len(post)
#                result = eval(code, _namespace) # mask on synapses
#                if result.dtype==float: # random number generation
#                    result=rand(len(post))<result
#                indexes=result.nonzero()[0]
#                presynaptic=pre[indexes]
#                postsynaptic=post[indexes]
#                dtype=self.synapses_pre[0].dtype
#                synapses_pre={}
#                nsynapses=0
#                for i in pre_slice:
#                    n=sum(result[i*len(post_slice):(i+1)*len(post_slice)])
#                    synapses_pre[i]=array(nsynapses+np.arange(n),dtype=dtype)
#                    nsynapses+=n
#            except MemoryError: # If not possible, vectorise over postsynaptic indexes
#                log_info("synapses","Construction of synapses cannot be fully vectorised (too big)")
            #del pre
            #del post
            #_namespace['i']=None
            #_namespace['j']=post_slice-post_shift
            #_namespace['n']=len(post_slice)
            synapses_pre={}
            nsynapses=0
            presynaptic,postsynaptic=[],[]
            for i in pre_slice:
                _namespace['i']=i-pre_shift # maybe an array rather than a scalar?
                result = eval(code, _namespace) # mask on synapses
                if result.dtype==float: # random number generation
                    result=rand(len(post_slice))<result
                indexes=result.nonzero()[0]
                n=len(indexes)
                synapses_pre[i]=np.array(nsynapses+np.arange(n),dtype=self.synapses_pre[0].dtype)
                presynaptic.append(i*np.ones(n,dtype=int))
                postsynaptic.append(post_slice[indexes])
                nsynapses+=n
                
            # Make sure the type is correct
            presynaptic=np.array(np.hstack(presynaptic),dtype=self.presynaptic.dtype)
            postsynaptic=np.array(np.hstack(postsynaptic),dtype=self.postsynaptic.dtype)
            synapses_post=None
        elif isinstance(value, np.ndarray):
            raise NotImplementedError
            nsynapses = np.array(value, dtype = int) 
            
        # Now create the synapses
        self.create_synapses(presynaptic,postsynaptic,synapses_pre,synapses_post)
    
    def create_synapses(self,presynaptic,postsynaptic,synapses_pre=None,synapses_post=None):
        '''
        Create new synapses.
        * synapses_pre: a mapping from presynaptic neuron to synapse indexes
        * synapses_post: same
        * presynaptic: an array of presynaptic neuron indexes (synapse->pre)
        * postsynaptic: same
        
        If synapses_pre or synapses_post is not specified, it is calculated from
        presynaptic or postsynaptic.       
        '''
        # Resize dynamic arrays and push new values
        newsynapses=len(presynaptic) # number of new synapses
        nvars,nsynapses_all=self._S.shape
        self._S.resize((nvars,nsynapses_all+newsynapses))
        self.presynaptic.resize(nsynapses_all+newsynapses)
        self.presynaptic[nsynapses_all:]=presynaptic
        self.postsynaptic.resize(nsynapses_all+newsynapses)
        self.postsynaptic[nsynapses_all:]=postsynaptic
        for delay_pre in self._delay_pre:
            delay_pre.resize(nsynapses_all+newsynapses)
        self._delay_post.resize(nsynapses_all+newsynapses)
        if synapses_pre is None:
            synapses_pre=invert_array(presynaptic,dtype=self.synapses_post[0].dtype)
        for i,synapses in synapses_pre.iteritems():
            nsynapses=len(self.synapses_pre[i])
            self.synapses_pre[i].resize(nsynapses+len(synapses))
            self.synapses_pre[i][nsynapses:]=synapses+nsynapses_all # synapse indexes are shifted
        if synapses_post is None:
            synapses_post=invert_array(postsynaptic,dtype=self.synapses_post[0].dtype)
        for j,synapses in synapses_post.iteritems():
            nsynapses=len(self.synapses_post[j])
            self.synapses_post[j].resize(nsynapses+len(synapses))
            self.synapses_post[j][nsynapses:]=synapses+nsynapses_all
    
    def __getattr__(self, name):
        if name == 'var_index':
            raise AttributeError
        if not hasattr(self, 'var_index'):
            raise AttributeError
        if (name=='delay_pre') or (name=='delay'): # default: delay is presynaptic delay
            if len(self._delay_pre)>1:
                return [SynapticDelayVariable(delay_pre,self,name) for delay_pre in self._delay_pre]
            else:
                return SynapticDelayVariable(self._delay_pre[0],self,name)
        elif name=='delay_post':
            return SynapticDelayVariable(self._delay_post,self,name)
        try:
            x=self.state(name)
            return SynapticVariable(x,self,name)
        except KeyError:
            return NeuronGroup.__getattr__(self,name)
        
    def __setattr__(self, name, val):
        if (name=='delay_pre') or (name=='delay'):
            if len(self._delay_pre)==1:
                SynapticDelayVariable(self._delay_pre[0],self,name)[:]=val
            else:
                raise NotImplementedError,"Cannot assign multiple delays at the same time"
        elif name=='delay_post':
            SynapticDelayVariable(self._delay_post,self,name)[:]=val
        else: # copied from Group
            origname = name
            if len(name) and name[-1] == '_':
                origname = name[:-1]
            if not hasattr(self, 'var_index') or (name not in self.var_index and origname not in self.var_index):
                object.__setattr__(self, name, val)
            else:
                if name in self.var_index:
                    x=self.state(name)
                else:
                    x=self.state_(origname)
                SynapticVariable(x,self,name).__setitem__(slice(None,None,None),val,level=2)
        
    def update(self): # this is called at every timestep
        '''
        Updates the synaptic variables.
        
        TODO:
        * Deal with static variables
        '''
        if self._state_updater is not None:
            self._state_updater(self)

        for queue, _namespace, code in zip(self.queues, self.namespaces, self.codes):
            synaptic_events = queue.peek()
            if len(synaptic_events):
                # Build the namespace - Here we don't consider static equations
                _namespace['_synapses'] = synaptic_events
                _namespace['t'] = self.clock._t
                exec code in _namespace
            queue.next()
            
    def connect_one_to_one(self,pre=None,post=None):
        '''
        Connects each neuron in the ``pre'' group to each corresponding one
        in the ``post'' group.
        '''
        if pre is None:
            pre = self.source
        if post is None:
            post = self.target
        pre, post = self.presynaptic_indexes(pre), self.postsynaptic_indexes(post)
        if len(pre) != len(post):
            raise TypeError,"Source and target groups do not have the same size"
            
        for i,j in zip(pre,post):
            self[i,j]=True
    
    def connect_random(self,pre=None,post=None,sparseness=None):
        '''
        Creates random connections between pre and post neurons
        (default: all neurons).
        This is equivalent to::
        
            S[pre,post]=sparseness
        
        ``pre=None''
            The set of presynaptic neurons, defined as an integer, an array, a slice or a subgroup.

        ``post=None''
            The set of presynaptic neurons, defined as an integer, an array, a slice or a subgroup.
        
        ``sparseness=None''
            The probability of connection of a pair of pre/post-synaptic neurons.
        '''
        if pre is None:
            pre=self.source
        if post is None:
            post=self.target
        pre,post=self.presynaptic_indexes(pre),self.postsynaptic_indexes(post)
        m=len(post)
        synapses_pre={}
        nsynapses=0
        presynaptic,postsynaptic=[],[]
        for i in pre: # vectorised over post neurons
            k = binomial(m, sparseness, 1)[0] # number of postsynaptic neurons
            synapses_pre[i]=nsynapses+np.arange(k)
            presynaptic.append(i*np.ones(k,dtype=int))
            # Not significantly faster to generate all random numbers in one pass
            # N.B.: the sample method is implemented in Python and it is not in Scipy
            postneurons = sample(xrange(m), k)
            #postneurons.sort() # sorting is unnecessary
            postsynaptic.append(post[postneurons])
            nsynapses+=k
        presynaptic=np.hstack(presynaptic)
        postsynaptic=np.hstack(postsynaptic)
        synapses_post=None # we ask for automatic calculation of (post->synapse)
        # this is more or less given by unique
        self.create_synapses(presynaptic,postsynaptic,synapses_pre,synapses_post)
        
    def presynaptic_indexes(self,x):
        '''
        Returns the array of presynaptic neuron indexes corresponding to x,
        which can be a integer, an array, a slice or a subgroup.
        '''
        return neuron_indexes(x,self.source)

    def postsynaptic_indexes(self,x):
        '''
        Returns the array of postsynaptic neuron indexes corresponding to x,
        which can be a integer, an array, a slice or a subgroup.
        '''
        return neuron_indexes(x,self.target)
    
    def compress(self):
        '''
        * Checks that the object is not empty.
        * Make the state array non-dynamical (important for the state updater).
        * Updates namespaces of pre and post code.
        '''
        if hasattr(self, '_iscompressed') and self._iscompressed:
            return
        self._iscompressed = True
        # Check that the object is not empty
        if len(self)==0:
            warnings.warn("Empty Synapses object")
        self._S=self._S[:,:]
        
        # Update namespaces of pre/post code        
        for _namespace in self.namespaces:
            for var,i in self.var_index.iteritems(): # no static variables here
                if isinstance(var, str):
                    _namespace[var]=self._S[i,:]
            for var,i in self.source.var_index.iteritems():
                if isinstance(var, str):
                    _namespace[var+'_pre']=self.source._S[i,:]
            for var,i in self.target.var_index.iteritems():
                if isinstance(var, str):
                    _namespace[var+'_post']=self.target._S[i,:]
                    _namespace[var]=self.target._S[i,:]
            _namespace['_pre']=self.presynaptic
            _namespace['_post']=self.postsynaptic
            _namespace['np']=np
            _namespace['binomial']=self._binomial
            _namespace['rand']=rand
            _namespace['randn']=randn
            _namespace['zeros']=np.zeros
            _namespace['sum']=sum
            
        self._iscompressed=True

    def synapse_index(self,i):
        '''
        Returns the synapse indexes correspond to i, which is a tuple.
        If i is a tuple (m,n), m and n can be an integer, an array, a slice or a subgroup.

        Searching synapse indexes for synapse (i,j) is implemented as follows.
        If i or j is an integer or a slice, they are converted to a boolean test.
        Then the following is executed:
        1) get indexes of target synapses of presynaptic neuron(s) i
        2) test whether postsynaptic neurons of these synapses correspond to j
        3) return synapses that passed the test
        or the symmetrical operations (depending on what is possible and faster).
        
        Otherwise, the following is executed:
        1) get indexes of target synapses of presynaptic neuron(s) i
        2) get indexes of source synapses of postsynaptic neuron(s) j
        3) calculate the intersection
        
        This will generally be ok for vectorised searches, but not for searching
        single elements (i,j). In this case, one might want to use
        a dictionary (i,j)->synapse index (not implemented). This is fast
        but 1) cannot be vectorised, 2) is very memory expensive.
        '''
        if not isinstance(i,tuple): # we assume it is directly a synapse index
            return i
        if len(i)==2:
            i,j=i
            # We use boolean tests if possible (faster)
            if isinstance(i,slice) or isinstance(i,int):
                test_i=slice_to_test(i)
            else:
                test_i=None
            if isinstance(j,slice) or isinstance(j,int):
                test_j=slice_to_test(j)
            else:
                test_j=None
            i=neuron_indexes(i,self.source)
            j=neuron_indexes(j,self.target)
            synapsetype=self.synapses_pre[0].dtype
            
            if (test_i is None) and (test_j is None): # no speed-up is possible
                synapses_pre=np.array(np.hstack([self.synapses_pre[k] for k in i]),dtype=synapsetype)
                synapses_post=np.array(np.hstack([self.synapses_post[k] for k in j]),dtype=synapsetype)
                return np.intersect1d(synapses_pre, synapses_post,assume_unique=True)
            elif ((len(i)<len(j)) and (test_j is not None)) or (test_i is None): # test synapses of presynaptic neurons
                synapses_pre=np.array(np.hstack([self.synapses_pre[k] for k in i]),dtype=synapsetype)
                return synapses_pre[test_j(self.postsynaptic[synapses_pre])]
            else: # test synapses of postsynaptic neurons
                synapses_post=np.array(np.hstack([self.synapses_post[k] for k in j]),dtype=synapsetype)
                return synapses_post[test_i(self.presynaptic[synapses_post])]
        elif len(i)==3: # 3rd coordinate is synapse number
            if np.isscalar(i[0]) and np.isscalar(i[1]):
                return self.synapse_index(i[:2])[i[2]]
            else:
                raise NotImplementedError,"The first two coordinates must be integers"
        return i
    
    def __repr__(self):
        return 'Synapses object with '+ str(len(self))+ ' synapses'

def smallest_inttype(N):
    '''
    Returns the smallest signed integer dtype that can store N indexes.
    '''
    if N<=127:
        return np.int8
    elif N<=32727:
        return np.int16
    elif N<=2147483647:
        return np.int32
    else:
        return np.int64

def indent(s,n=1):
    '''
    Inserts an indentation (4 spaces) or n before the multiline string s.
    '''
    return re.compile(r'^',re.M).sub('    '*n,s)

def invert_array(x,dtype=int):
    '''
    Returns a dictionary y of N int arrays such that:
    y[i]=set of j such that x[j]==i
    '''
    if len(x) == 0:
        return {}

    I = np.argsort(x) # ,kind='mergesort') # uncomment for a stable sort
    xs = x[I]
    # This below does the same as unique, except the indices point to first time
    # each number appears in the array 
    # See also code for unique (doesn't use diff, not sure which one is faster)
    indices=np.hstack(([0],np.where(np.diff(xs)!=0)[0]+1)) # or concatenate?
    u=xs[indices]
    y={}
    for j,i in enumerate(u[:-1]):
        y[i]=np.array(I[indices[j]:indices[j+1]],dtype=dtype)
    y[u[-1]]=np.array(I[indices[-1]:],dtype=dtype)
    return y


def neuron_indexes(x,P):
    '''
    Returns the array of neuron indexes corresponding to x,
    which can be a integer, an array, a slice or a subgroup.
    P is the neuron group.
    '''
    if isinstance(x,NeuronGroup): # it should be checked that x is actually a subgroup of P
        i0=x._origin - P._origin # offset of the subgroup x in P
        return np.arange(i0,i0+len(x))
    else:
        return slice_to_array(x,N=len(P))      


def slice_to_test(x):
    '''
    Returns a testing function corresponding to whether an index is in slice x.
    x can also be an int.
    '''
    if isinstance(x,int):
        return lambda y:y==x
    elif isinstance(x,slice):
        start,stop,step=x.start,x.stop,x.step
        if start is None:
            start=0
        if step is None:
            step=1
        if stop is None:
            return lambda y:(y>=start) & ((y-start)%step==0)
        else:
            return lambda y:(y>=start) & (y<stop) & ((y-start)%step==0)

if __name__=='__main__':
    #log_level_debug()
    print invert_array(np.array([7,5,2,2,3,5]))