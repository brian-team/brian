'''
The Synapses class - see BEP-21

TODO:
* CUBA and speed test
* Do the TODOs
* setitem and getattr (includes a special vector class with synaptic access)

TODO (later):
* State updates and event-driven stuff
* Max delay should be calculated at run time (compress)
* Replace spike queue data with a dynamic array object
'''
from brian import *
from brian.utils.dynamicarray import *
from spikequeue import *
import numpy as np
from brian.inspection import *
from brian.equations import *
from brian.utils.documentation import flattened_docstring
import re

__all__ = ['Synapses']

class Synapses(NeuronGroup): # This way we inherit a lot of useful stuff
    def __init__(self, source, target = None, model = None, 
             max_delay = 0, # is this useful?
             level = 0,
             clock = None,
             unit_checking = True, method = None, freeze = False, implicit = False, order = 1, # model (state updater) related
             pre = '', post = ''):
        N=len(source) # initial number of synapses = number of presynaptic neurons
        self.source=source
        self.target=target or source # default is target=source

        # Check clocks. For the moment we enforce the same clocks for all objects
        clock = clock or source.clock
        if source.clock!=target.clock:
            raise ValueError,"Source and target groups must have the same clock"

        NeuronGroup.__init__(self, 1,model=model,clock=clock,level=level+1,unit_checking=unit_checking,method=method,freeze=freeze,implicit=implicit,order=order)
        # We might want *not* to use the state updater on all variables, so for now I disable it (see update())
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

        # Pre and postsynaptic indexes
        self.presynaptic=DynamicArray(len(self),dtype=smallest_inttype(len(self.source))) # this should depend on number of neurons
        self.postsynaptic=DynamicArray(len(self),dtype=smallest_inttype(len(self.target))) # this should depend on number of neurons

        # Pre and postsynaptic delays (synapse -> delay)
        self.delay_pre=DynamicArray(len(self),dtype=int16) # max 32767 delays
        self.delay_post=DynamicArray(len(self),dtype=int16)
        
        # Pre and postsynaptic synapses (i->synapse indexes)
        max_synapses=4294967296 # it could be explicitly reduced by a keyword
        # dictionaries could be more efficient
        self.synapses_pre=[DynamicArray(0,dtype=smallest_inttype(max_synapses))]*len(self.source) # list of dynamic arrays
        self.synapses_post=[DynamicArray(0,dtype=smallest_inttype(max_synapses))]*len(self.target)

        self.generate_code(pre,post,level+1) # I moved this in a separate method to clarify the init code
        
        # Event queues
        self.pre_queue = SpikeQueue(self.source, self, max_delay = max_delay)
        #self.post_queue = SpikeQueue(self.target, self, max_delay = max_delay)

        self.contained_objects = [self.pre_queue]
      
    def generate_code(self,pre,post,level):
        '''
        Generates pre and post code.
        For the moment, we only deal with pre code.
        
        TODO:
        * post code
        * include static variables
        * have a list of variable names
        * deal with v_post, v_pre
        '''
        # Handle multi-line pre, post equations and multi-statement equations separated by ;
        # (this should probably be factored)
        if '\n' in pre:
            pre = flattened_docstring(pre)
        elif ';' in pre:
            pre = '\n'.join([line.strip() for line in pre.split(';')])
        if '\n' in post:
            post = flattened_docstring(post)
        elif ';' in post:
            post = '\n'.join([line.strip() for line in post.split(';')])
        
        # Create namespaces
        pre_namespace = namespace(pre, level = level + 1)
        pre_namespace['target'] = self.target # maybe we could save one indirection here
        pre_namespace['unique'] = np.unique
        pre_namespace['nonzero'] = np.nonzero
        #pre_namespace['_pre'] = self._statevector._pre # mapping synapse -> pre (change to dynamic array)
        #pre_namespace['_post'] = self._statevector._post
        #varnames=self.units.keys() # not useful because it is done at run time
        #for var in varnames:
        #    pre_namespace[var] = self._state(var)

        # Generate the code
        def update_code(pre, indices):
            res = pre
            # given the synapse indices, write the update code,
            # this is here because in the code we generate we need to write this twice (because of the multiple presyn spikes for the same postsyn neuron problem)
                       
            # Replace synaptic variables by their value
            for var in self.var_index: # static variables are not included here
                if isinstance(var, str):
                    res = re.sub(r'\b' + var + r'\b', var + '['+indices+']', res) # synaptic variable, indexed by the synapse number
 
            # Replace postsynaptic variables by their value
            for postsyn_var in self.target.var_index: # static variables are not included here
                if isinstance(postsyn_var, str):
                    res = re.sub(r'\b' + postsyn_var + r'\b', 'target.' + postsyn_var + '[_post['+indices+']]', res)# postsyn variable, indexed by post syn neuron numbers
 
            return res
 
        # pre code
        pre_code = ""
        pre_code += "_u, _i = unique(_post, return_index = True)\n"
        pre_code += update_code(pre, '_i') + "\n"
        pre_code += "if len(_u) < len(_post):\n"
        pre_code += "    _post[_i] = -1\n"
        pre_code += "    while (len(_u) < len(_post)) & (_post>-1).any():\n" # !! the any() is time consuming (len(u)>=1??)
        pre_code += "        _u, _i = unique(_post, return_index = True)\n"
        pre_code += "        " + update_code(pre, '_i[1:]') + "\n"
        pre_code += "        _post[_i[1:]] = -1 \n"
        log_debug('brian.synapses', '\nPRE CODE:\n'+pre_code)
                
        # Commpile
        pre_code = compile(pre_code, "Presynaptic code", "exec")
        
        self.pre_namespace = pre_namespace
        self.pre_code = pre_code

    def __setitem__(self, key, value):
        '''
        Creates new synapses.
        Caution:
        1) there is no deletion
        2) synapses are added, not replaced (e.g. S[1,2]=True;S[1,2]=True creates 2 synapses)
        '''
        if not isinstance(key, tuple): # we should check that number of elements is 2 as well
            raise ValueError('Synapses behave as 2-D objects')
        pre,post=key # pre and post indexes (can be slices)
        
        if isinstance(value, (int, bool)): # ex. S[1,7]=True
            # Simple case, either one or multiple synapses between different neurons
            if value is False:
                raise ValueError('Synapses can be deleted')
            elif value is True:
                nsynapses = 1
            else:
                nsynapses = value
        
            # We need to create:
            # 1) a list of presynaptic indexes (synapse -> pre)
            # 2) a list of postsynaptic indexes (synapse -> post)
            # 3) a mapping pre -> synapse (dictionary)
            # 4) a mapping post -> synapse
            #
            # meshgrid(pre,post) returns (synapse -> pre (array) and synapse -> post)
        
            pre_slice = slice_to_array(pre)
            post_slice = slice_to_array(post)
            presynaptic,postsynaptic=meshgrid(pre,post) # synapse -> pre, synapse -> post
            # TODO: pre,post -> synapse
        
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
        return NeuronGroup.__getattr__(self,name)
        
    def update(self): # this is called at every timestep
        '''
        TODO:
        * Have namespaces partially built at run time (call state_(var)),
          or better, extract synaptic events from the synaptic state matrix;
          same stuff for postsynaptic variables
        * Deal with static variables
        * Execute post code
        * Insert lastspike
        * Insert rand()
        '''
        #NeuronGroup.update(self) # we don't do it for now
        synaptic_events = self.pre_queue.peek()
        if len(synaptic_events):
            print synaptic_events
            # Build the namespace - Maybe we should do this only once? (although there is the problem of static equations)
            # Careful: for dynamic arrays you need to get fresh references at run time
            _namespace = self.pre_namespace
            #_namespace['_synapses'] = synaptic_events # not needed any more
            for var in self.var_index: # in fact I should filter out integers ; also static variables are not included here
                _namespace[var] = self.state_(var)[synaptic_events] # could be faster to directly extract a submatrix from _S
            _namespace['t'] = self.clock._t
            _namespace['_pre']=self.presynaptic[synaptic_events].copy() # could be faster to have an array already there
            _namespace['_post']=self.postsynaptic[synaptic_events].copy()
            exec self.pre_code in _namespace
        
        self.pre_queue.next()
        #self.post_queue.next()
    
    def __repr__(self):
        return 'Synapses object with '+ str(len(self))+ ' synapses'

def smallest_inttype(N):
    '''
    Returns the smallest dtype that can store N indexes
    '''
    if N<=127:
        return int8
    elif N<=32727:
        return int16
    elif N<=2147483647:
        return int32
    else:
        return int64

def slice_to_array(s):
    '''
    Converts a slice s or single int to the corresponding array of integers
    '''
    if isinstance(s,slice):
        arange(s.start,s.stop,s.end)
    else: # if not a slice (e.g. an int) then we return it as an array of a single element
        return array(s)

if __name__=='__main__':
    log_level_debug()
    P=NeuronGroup(2,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
    Q=NeuronGroup(1,model='v:1')
    S=Synapses(P,Q,model='w:1',pre='v+=w')
    M=StateMonitor(Q,'v',record=True)
    
    S.synapses_pre[0]=[0]
    S.synapses_pre[1]=[1]
    S.delay_pre[0]=10 # in time bins
    S.delay_pre[1]=50
    S.presynaptic[0]=0
    S.presynaptic[1]=1
    S.w[0]=1.
    S.w[1]=0.
    
    run(50*ms)

    plot(M.times/ms,M[0])
    # doesn't work! Delays not taken into account
    
    show()
