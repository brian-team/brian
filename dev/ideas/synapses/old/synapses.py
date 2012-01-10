'''
Here is defined the main Synapses code.

Two important things:

Synapse object

EventSynapseUpdater

I tried to write as much as I could in the classdoc of each object.

'''

from brian import *
import numpy as np
from brian.inspection import *
from brian.equations import *
from brian.utils.documentation import flattened_docstring
from dev.ideas.synapses.statevectors import *
from dev.ideas.synapses.spikequeue import *
import re

__all__ = ['Synapses']

class Synapses(NetworkOperation):
    '''
    Synapses object
   
     State:
    - the model kwdarg can only declare variables for now
    - only the 'pre' code is implemented 
    - multiple synapses (between same pair of neurons), with multiple delays seem to work fine!
    
    TODO:
    - postsynaptic delays


    ** Initialization **
    
    synapses = syn.Synapses(gin, gout, model = 'w : 1', pre = 'v += w', post = '')
    
    * Arguments * 
    
    ``gin'' source NeuronGroup
    ``gout'' target NeuronGroup
    
    * Keywords *
    
    ``model'' for now only declares variables
    ``pre'' code to be executed when a presynaptic spike arrives
    ``post'' code to be executed when a postsynaptic spike arrives
    
    ``max_delay'' If not set then the synapse is without delays, otherwise sets the maximum acceptable delay for the Synapses. Note that the delays are only axonal (i.e. presynaptic), this could easily be changed in the future.
    
    
    * Attributes *
    
    All the variables of the model (see below), plus
    
    ``existing_synapses'' a 2 D array of shape (2, len(synapses)) that hold the pre and post neuron numbers for each synapse.
    
    
    * Adding Synapses *
    
    When it's created, a Synapses object doesn't have any instantiated synapses. To add some, one must do:
    
    synapses[:,0] = True
    
    Which connects all the presynaptic neurons to the postsynaptic neuron
    
    Alternatively, one may want to add 2 synapses between the same neurons, to do so, one can run the above code line a few times, of just write 
    synapses[:,0] = 2
   
    * Setting variables * 
    
    All of the variables defined in the model code (except the reserved ones), can (and should) be instantiated. In the above example, to set the weights, one can do:
    synapses.w[:,0] = 1 (or array, etc...)
    
    To set all the weights of the *EXISTING* synapses to 1.
    
    If there are multiple synapses, then they can be accessed with a third field:
    synapses.w[:,0,1] = 2
    By default the previous call sets all synapses to the same value.
    
    * Delays * 
    
    If the delays are activated (setting the kwdarg max_delay) then a field of delays (``delay'') is created and can be used as any other field by setting it:

    synapses.delay[:,0] = 5*ms

    '''
    def __init__(self, source, target = None, 
                 clock = None,
                 max_delay = 0,
                 level = 0, model = None,
                 check_units = True, method = None, freeze = False, implicit = False, order = 1, # model (state updater) related
                 pre = '', post = ''):
        
        NetworkOperation.__init__(self, lambda:None, clock=clock)

        # Arguments parsing
        if target is None:
            self.source = self.target = source
        else:
            self.source = source
            self.target = target
        

        ########### CODE PARSING
        # model equations parsing
        if isinstance(model, Equations):
            model_obj = model
        else:
            if '\n' in model:
                model = flattened_docstring(model)
            elif ';' in model:
                model = '\n'.join([line.strip() for line in model.split(';')])
            model_obj = Equations(model, level = level + 1)
            
        # stolen from NeuronGroup
        # !! This will not work with postsynaptic variables in the model eqs !!
        if isinstance(model_obj, StateUpdater):
            self._state_updater = model_obj
            self._all_units = defaultdict() # what is that
        elif isinstance(model_obj, Equations):
            self._eqs = model_obj
            self._state_updater, var_names = magic_state_updater(model_obj, clock=clock, order=order,
                                                                 check_units=check_units, implicit=implicit,
                                                                 compile=compile, freeze=freeze,
                                                                 method=method)
            self._state_updater_varnames = var_names
            
            
        # TODO: 
        # - Maybe stateupdaters should be aware of the dtype of the data (or else my state vectors thing is sort of useless
        # - or ensure fast access to different data types in a vectorized way?
            
        # handle multi-line pre, post equations and multi-statement equations separated by ;
        if '\n' in pre:
            pre = flattened_docstring(pre)
        elif ';' in pre:
            pre = '\n'.join([line.strip() for line in pre.split(';')])
        if '\n' in post:
            post = flattened_docstring(post)
        elif ';' in post:
            post = '\n'.join([line.strip() for line in post.split(';')])

        # Check units
        model_obj.compile_functions()
        #model_obj.check_units()
        # Get variable names
        self.vars = model_obj._diffeq_names # !! there are also static variables and aliases !!
        

        ############# Setting up the data structure
        # Mandatory fields: 3 for pre/post/delay_pre, all int32 TODO: finer pick of dtype!
        
        # REPLACE THAT BY MORE SENSIBLE DATA STRUCTURE
        dtypes = (np.int32, ) * 3
        default_labels = ['_pre', '_post', 'delay']
        # Equation defined fields (float32)
        dtypes += (np.float32, ) * len(self.vars)

        # construction of the structure
        self._statevector = ConstructionSparseStateVector(len(dtypes), dtype = dtypes, labels = default_labels+self.vars)
        

        ############# Code!!!
        # create namespace
        pre_namespace = namespace(pre, level = level + 1)
        pre_namespace['target'] = self.target
        pre_namespace['unique'] = np.unique
        pre_namespace['nonzero'] = np.nonzero
        pre_namespace['_pre'] = self._statevector._pre
        pre_namespace['_post'] = self._statevector._post

        for var in self.vars:
            pre_namespace[var] = self._statevector[var]
        # !! also add postsynaptic variables !!

        def update_code(pre, indices):
            res = pre
            # given the synapse indices, write the update code,
            # this is here because in the code we generate we need to write this twice (because of the multiple presyn spikes for the same postsyn neuron problem)
            for postsyn_var in self.target.var_index:
                if isinstance(postsyn_var, str):
                    res = re.sub(r'\b' + postsyn_var + r'\b', 'target.' + postsyn_var + '[_post['+indices+']]', res)# postsyn variable, indexed by post syn neuron numbers
                
            for var in self.vars:
                res = re.sub(r'\b' + var + r'\b', var + '['+indices+']', res) # synaptic variable, indexed by the synapse number
            return res
 
        # pre code
        pre_code = "_post_neurons = _post[_synapses]\n"
        # which post syn neurons
        pre_code += "_u, _i = unique(_post_neurons, return_index = True)\n"
        pre_code += update_code(pre, '_synapses[_i]') + "\n"
        pre_code += "if len(_u) < len(_post_neurons):\n"
        pre_code += "    _post_neurons[_i] = -1\n"
        pre_code += "    while (len(_u) < len(_post_neurons)) & (_post_neurons>-1).any():\n" # !! the any() is time consuming (len(u)>=1??)
        pre_code += "        _u, _i = unique(_post_neurons, return_index = True)\n"
        pre_code += "        " + update_code(pre, '_synapses[_i[1:]]') + "\n"
        pre_code += "        _post_neurons[_i[1:]] = -1 \n"
        log_debug('brian.synapses', '\nPRE CODE:\n'+pre_code)
        
        pre_code = compile(pre_code, "Presynaptic code", "exec")
        
        
        self.pre_namespace = pre_namespace
        self.pre_code = pre_code
        self.pre_queue = SpikeQueue(self.source, self, max_delay = max_delay)

        self.contained_objects = [self.pre_queue] # wtf is this for: so that pre_queue.propagate is called
        
        
        self._pre_to_synapse = np.zeros(len(self.source), dtype = object)
        self._post_to_synapse = np.zeros(len(self.target), dtype = object)
        # presyn neuron to postsyn neuron mapping
        
    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            # do they?
            raise ValueError('Synapses behave as 2-D objects')

        if isinstance(value, (int, bool)):
            # Simple case, either one or multiple synapses between different neurons
            if not (value is True):
                n_synapses = value
            else: # !! and False? !!
                n_synapses = 1
        
            pre_slice = slice2range(key[0], len(self.source))
            post_slice = slice2range(key[1], len(self.target))
        
            # Construction of the pre/post neurons indices list
            # This needs speed up!!!!
            # BTW not sure that lists are really necessary, maybe at build time though
            pre, post = [], []
            for i in post_slice:
                post += [i]*len(pre_slice)*n_synapses 
                pre += list(pre_slice)*n_synapses

        elif isinstance(value, str):
            # !! Other more efficient option for memory consumption: !!
            # vectorize only over post neurons, loop over pre
            # transform boolean to sparse vector after every pre neuron
            #
            # more complex case where the value is specified as a string, hence code
            # The code is evaluated and this yields a matrix that is then passed 
            code = re.sub(r'\b' + 'rand\(\)', 'rand(len(_i), len(_j))', value) # replacing rand()

            pre_slice = slice2range(key[0], len(self.source))
                
            post_slice = slice2range(key[1], len(self.target))

            pre_grid, post_grid = np.meshgrid(pre_slice, post_slice)
            
            namespace = {'_i' : pre_slice,
                         '_j' : post_slice,
                         'i' : pre_grid,
                         'j' : post_grid,
                         'rand': np.random.rand}
            log_debug('synapses.setitem', code)
            
            res = eval(code, namespace)
            
            log_debug('synapses.setitem', 'created synapses: \n '+ str(res))
            
            self[key] = res
            return
        elif isinstance(value, np.ndarray):
            n_syn = array(value.copy(), dtype = int)
            pre, post = np.nonzero(n_syn)
            if (n_syn >= 1).any():
                # no need to go through this if there is never multiple synapses
                pre = list(pre)
                post = list(post)
                n_syn -= 1
                n_syn[n_syn < 0] = 0
                _pre, _post = np.nonzero(n_syn)
                while len(_pre) != 0:
                    pre += list(_pre)
                    post += list(_post)
                    n_syn -= 1
                    n_syn[n_syn < 0] = 0
                    _pre, _post = np.nonzero(n_syn)
        # !! add: sparse matrices
        values = np.vstack((pre, post))
        
        # append values and get initial index (for the _pre_to_synapse struct)
        initial_index = self._statevector.append(values) 
        
        # build the synapse reference arrays
        for i in np.unique(values[0,:]):
            newindices = initial_index + np.nonzero(values[0,:] == i)[0]
            self._pre_to_synapse[i] = np.hstack((self._pre_to_synapse[i], 
                                                 newindices))

        for i in np.unique(values[1,:]):
            newindices = initial_index + np.nonzero(values[1,:] == i)[0]
            self._post_to_synapse[i] = np.hstack((self._post_to_synapse[i], 
                                                 newindices))
        
    def __getattr__(self, name):
        if hasattr(self, '_statevector'): # always True?
            if name == '_S':
                # for the state updater, I should do this better I guess
                return self._statevector._allstates
            if name in self._statevector.labels:
                # Have to return a special kind of Vector, for the slicing to work
                data = getattr(self._statevector, name)
                groups_shape = (len(self.source), len(self.target))

                # dt handling (see the ParameterVector doc)
                dt = None
                if name == 'delay':
                    dt = float(self.source.clock.dt)

                return ParameterVector(data, self, delay_dt = dt)
        try:
            self.__dict__[name]
        except KeyError:
            raise AttributeError('Synapses object doesn\'t have a '+name+' attribute')

    def __len__(self):
        '''
        Returns the number of existing synapses.
        '''
        return self._statevector.nvalues
    
    def __call__(self):
        # presynaptic spikes
        synapses_current = self.pre_queue.peek()
        if len(synapses_current):
            _namespace = self.pre_namespace
            _namespace['_synapses'] = np.array(synapses_current)
            for var in self.vars:
                _namespace[var] = self._statevector[var]
            _namespace['t'] = self.source.clock._t
            _namespace['_post'] = self._statevector._post
            exec self.pre_code in _namespace
            
        self.pre_queue.next()
        
        # state update
        self._state_updater(self)
        
    def pre2synapses(self, presyn):
        return np.hstack(self._pre_to_synapse[presyn])

    def post2synapses(self, postsyn):
        return np.hstack(self._pre_to_synapse[postsyn])

    @property
    def existing_synapses(self):
        return np.vstack((self._statevector._pre, self._statevector._post))
    
    def __repr__(self):
        s = 'Synapses:\n'
        s+= str(self.existing_synapses)
        return s

# DEPRECATED STUFF:
#
# ########################## SynapseUpdater ##########################

# class EventSynapseUpdater(SpikeMonitor):
#     '''
#     Mimicked after the eventstdpupdater in dev/ideas/stdp/eventdriven_stdp
    
#     A SpikeMonitor that executes the code it's given at initialization.
    
#     The interesting code is rather in the Synapses object itself (search for pre_code)
#     '''
#     def __init__(self, source, S, code, namespace, delay=0 * ms):
#         '''
#         source = source group
#         S = base Synapse object
#         vars = variable names
#         M = matrix of the linear differential system
#         code = code to execute for every spike
#         namespace = namespace for the code
#         delay = transmission delay 
#         '''
#         super(EventSynapseUpdater, self).__init__(source, 
#                                                   record = False)
#         self._code = code # update code
#         self._namespace = namespace # code namespace
#         self.S = S
        
#     def propagate(self, spikes):
#         if len(spikes):
#             synapses = []
#             for i in spikes:
#                 synapses += list(nonzero(self.S._statevector._pre == i)[0])
                
#             self._namespace['_synapses'] = np.array(synapses)
#             for var in self.S.vars:
#                 self._namespace[var] = self.S._statevector[var]
#             self._namespace['t'] = self.S.source.clock._t
#             self._namespace['_post'] = self.S._statevector._post
#             exec self._code in self._namespace


# INTIAL_MAXSPIKESPER_DT = 10 # Because SpikeQueue doesn't have a dynamic structure (yet)
# class DelayEventSynapseUpdater(SpikeMonitor):
#     '''
#     Same as above but includes delays (yay!)
#     '''
#     def __init__(self, source, S, code, namespace, max_delay):
#         '''

#         source = source group
#         S = base Synapse object
#         vars = variable names
#         M = matrix of the linear differential system
#         code = code to execute for every spike
#         namespace = namespace for the code

#         max_delay = maximum delay
#         '''
#         super(DelayEventSynapseUpdater, self).__init__(source, 
#                                                   record = False)
#         self._code = code # update code
#         self._namespace = namespace # code namespace
#         self.S = S
#         # SpikeQueue initialization
#         nsteps = int(np.ceil((max_delay)/(self.S.source.clock.dt)))
#         self.spike_queue = SpikeQueue(source, INTIAL_MAXSPIKESPER_DT)
        
#     def propagate(self, spikes):
#         # step 1: insert spikes in the event queue           
#         if len(spikes):
#             # synapse identification, 
#             # this seems ok in terms of speed even though I dont like the for loop. 
#             # any idea? see stest_fastsynapseidentification.py
#             synapses = []
#             for i in spikes:
#                 synapses += list(nonzero(self.S._statevector._pre == i)[0]) 


#             # delay getting:
#             delays = self.S._statevector.delay[synapses]
#             offsets = self.spike_queue.offsets(delays)
#             self.spike_queue.insert(delays, offsets, synapses)
            
#         # step 2: execute code on current spikes
#         synapses_current = self.spike_queue.peek()
#         if len(synapses_current):
#             self._namespace['_synapses'] = np.array(synapses_current)
#             for var in self.S.vars:
#                 self._namespace[var] = self.S._statevector[var]
#             self._namespace['t'] = self.S.source.clock._t
#             self._namespace['_post'] = self.S._statevector._post
#             exec self._code in self._namespace
            
#         self.spike_queue.next()
