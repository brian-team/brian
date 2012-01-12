'''
The Synapses class - see BEP-21

TODO:
* Unknown bug somewhere!
* Faster queue
* Automatically precompute offsets
* getattr (includes a special vector class with synaptic access)
* Do the TODOs
* setitem

TODO (later):
* State updates and event-driven stuff
* Max delay should be calculated at run time (compress)
* Replace spike queue data with a dynamic array object?
* Replace NeuronGroup.__init__ with own stuff
'''
from brian import *
from brian.utils.dynamicarray import *
from spikequeue import *
import numpy as np
from brian.inspection import *
from brian.equations import *
from numpy.random import binomial
from brian.utils.documentation import flattened_docstring
from random import sample
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
        target=target or source # default is target=source
        self.source=source
        self.target=target

        # Check clocks. For the moment we enforce the same clocks for all objects
        clock = clock or source.clock
        if source.clock!=target.clock:
            raise ValueError,"Source and target groups must have the same clock"

        NeuronGroup.__init__(self, 0,model=model,clock=clock,level=level+1,unit_checking=unit_checking,method=method,freeze=freeze,implicit=implicit,order=order)
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
        # _S is turned to a dynamic array - OK this is probably not good! we may lose references at this point
        S=self._S
        self._S=DynamicArray(S.shape)
        self._S[:]=S

        # Pre and postsynaptic indexes (synapse -> pre/post)
        self.presynaptic=DynamicArray(len(self),dtype=smallest_inttype(len(self.source))) # this should depend on number of neurons
        self.postsynaptic=DynamicArray(len(self),dtype=smallest_inttype(len(self.target))) # this should depend on number of neurons

        # Pre and postsynaptic delays (synapse -> delay_pre/delay_post)
        self.delay_pre=DynamicArray(len(self),dtype=int16) # max 32767 delays
        self.delay_post=DynamicArray(len(self),dtype=int16)
        
        # Pre and postsynaptic synapses (i->synapse indexes)
        max_synapses=4294967296 # it could be explicitly reduced by a keyword
        # We use a loop instead of *, otherwise only 1 dynamic array is created
        self.synapses_pre=[DynamicArray(0,dtype=smallest_inttype(max_synapses)) for _ in range(len(self.source))]
        self.synapses_post=[DynamicArray(0,dtype=smallest_inttype(max_synapses)) for _ in range(len(self.target))]
        # Turn into dictionaries?
        #self.synapses_pre=dict(enumerate(synapses_pre))
        #self.synapses_post=dict(enumerate(synapses_post))

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
        Synapse indexes are created such that synapses with the same presynaptic neuron
        and delay have contiguous indexes.
        
        Caution:
        1) there is no deletion
        2) synapses are added, not replaced (e.g. S[1,2]=True;S[1,2]=True creates 2 synapses)
        
        TODO:
        * S[:,:]='i<j'
        * S[:,:]=array (boolean or int)
        * S.connect_random
        * S[1,[2,5,7]]=True
        * S[[0,1],[5,9]]=True # same as in numpy (creates 0->5 and 1->9)
        '''
        if not isinstance(key, tuple): # we should check that number of elements is 2 as well
            raise ValueError('Synapses behave as 2-D objects')
        pre,post=key # pre and post indexes (can be slices)
        
        '''
        Each of these sets of statements creates:
        * synapses_pre: a mapping from presynaptic neuron to synapse indexes
        * synapses_post: same
        * presynaptic: an array of presynaptic neuron indexes (synapse->pre)
        * postsynaptic: same
        '''
        if isinstance(value, (int, bool)): # ex. S[1,7]=True
            # Simple case, either one or multiple synapses between different neurons
            if value is False:
                raise ValueError('Synapses can be deleted')
            elif value is True:
                nsynapses = 1
            else:
                nsynapses = value

            pre_slice = self.presynaptic_indexes(pre)
            post_slice = self.postsynaptic_indexes(post)
            # Bound checks
            if pre_slice[-1]>=len(self.source):
                raise ValueError('Presynaptic index greater than number of presynaptic neurons')
            if post_slice[-1]>=len(self.target):
                raise ValueError('Postsynaptic index greater than number of postsynaptic neurons')
            postsynaptic,presynaptic=meshgrid(post_slice,pre_slice) # synapse -> pre, synapse -> post
            # Flatten
            presynaptic.shape=(presynaptic.size,)
            postsynaptic.shape=(postsynaptic.size,)
            # pre,post -> synapse index, relative to last synapse
            # (that's a complex vectorised one!)
            synapses_pre=arange(len(presynaptic)).reshape((len(pre_slice),len(post_slice)))
            synapses_post=ones((len(post_slice),1),dtype=int)*arange(0,len(presynaptic),len(post_slice))+\
                          arange(len(post_slice)).reshape((len(post_slice),1))
            # Repeat
            if nsynapses>1:
                synapses_pre=hstack([synapses_pre+k*len(presynaptic) for k in range(nsynapses)]) # could be vectorised
                synapses_post=hstack([synapses_post+k*len(presynaptic) for k in range(nsynapses)]) # could be vectorised
                presynaptic=presynaptic.repeat(nsynapses)
                postsynaptic=postsynaptic.repeat(nsynapses)
            # Turn into dictionaries
            synapses_pre=dict(zip(pre_slice,synapses_pre))
            synapses_post=dict(zip(post_slice,synapses_post))
        
        # Now create the synapses
        self.create_synapses(presynaptic,postsynaptic,synapses_pre,synapses_post)
    
    def create_synapses(self,presynaptic,postsynaptic,synapses_pre,synapses_post=None):
        '''
        Create new synapses.
        * synapses_pre: a mapping from presynaptic neuron to synapse indexes
        * synapses_post: same
        * presynaptic: an array of presynaptic neuron indexes (synapse->pre)
        * postsynaptic: same
        
        TODO:
        * option to automatically create postsynaptic from synapses_post
        '''
        # Resize dynamic arrays and push new values
        newsynapses=len(presynaptic) # number of new synapses
        nvars,nsynapses=self._S.shape
        self._S.resize((nvars,nsynapses+newsynapses))
        self.presynaptic.resize(nsynapses+newsynapses)
        self.presynaptic[nsynapses:]=presynaptic
        self.postsynaptic.resize(nsynapses+newsynapses)
        self.postsynaptic[nsynapses:]=postsynaptic
        self.delay_pre.resize(nsynapses+newsynapses)
        self.delay_post.resize(nsynapses+newsynapses)
        for i,synapses in synapses_pre.iteritems():
            nsynapses=len(self.synapses_pre[i])
            self.synapses_pre[i].resize(nsynapses+len(synapses))
            self.synapses_pre[i][nsynapses:]=synapses+nsynapses # synapse indexes are shifted
        if synapses_post is not None:
            for j,synapses in synapses_post.iteritems():
                nsynapses=len(self.synapses_post[j])
                self.synapses_post[j].resize(nsynapses+len(synapses))
                self.synapses_post[j][nsynapses:]=synapses+nsynapses
    
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
        
    def connect_random(self,pre=None,post=None,sparseness=None):
        '''
        Creates random connections between pre and post neurons
        (default: all neurons).
        '''
        pre=pre or self.source
        post=post or self.target
        pre,post=self.presynaptic_indexes(pre),self.postsynaptic_indexes(post)
        m=len(post)
        synapses_pre={}
        nsynapses=0
        presynaptic,postsynaptic=[],[]
        for i in pre: # vectorised over post neurons
            k = binomial(m, sparseness, 1)[0] # number of postsynaptic neurons
            synapses_pre[i]=nsynapses+arange(k)
            presynaptic.append(i*ones(k,dtype=int))
            # Not significantly faster to generate all random numbers in one pass
            # N.B.: the sample method is implemented in Python and it is not in Scipy
            postneurons = sample(xrange(m), k)
            postneurons.sort()
            postsynaptic.append(postneurons)
            nsynapses+=k
        presynaptic=hstack(presynaptic)
        postsynaptic=hstack(postsynaptic)
        synapses_post=None # we ask for automatic calculation of (post->synapse)
        # this is more or less given by unique
        self.create_synapses(presynaptic,postsynaptic,synapses_pre,synapses_post)
        '''
        TODO NOW:
        * automatic calculation of post->synapse
        '''
        
    def presynaptic_indexes(self,x):
        '''
        Returns the array of presynaptic neuron indexes corresponding to x,
        which can be a integer, an array or a subgroup
        '''
        return neuron_indexes(x,self.source)

    def postsynaptic_indexes(self,x):
        '''
        Returns the array of postsynaptic neuron indexes corresponding to x,
        which can be a integer, an array or a subgroup
        '''
        return neuron_indexes(x,self.target)
    
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

def slice_to_array(s,N=None):
    '''
    Converts a slice s or single int to the corresponding array of integers.
    N is the maximum number of elements, this is used to handle negative numbers
    in the slice.
    '''
    if isinstance(s,slice):
        start=s.start or 0
        stop=s.stop or N
        step=s.step
        if stop<0 and N is not None:
            stop=N+stop
        return arange(start,stop,step)
    else: # if not a slice (e.g. an int) then we return it as an array of a single element
        return array([s])

def neuron_indexes(x,P):
    '''
    Returns the array of neuron indexes corresponding to x,
    which can be a integer, an array or a subgroup.
    P is the neuron group.
    '''
    if isinstance(x,NeuronGroup): # it should be checked that x is actually a subgroup of P
        i0=x._origin - P._origin # offset of the subgroup x in P
        return arange(i0,i0+len(x))
    else:
        return slice_to_array(x,N=len(P))      

if __name__=='__main__':
    #log_level_debug()
    P=NeuronGroup(10,model='v:1')
    S=Synapses(P,model='w:1')
    S[0:2,0:3]=True
    #S[1,2]=True
    S.w=.5
    print S.w
    print S.presynaptic[0]
    print S.synapses_pre[1]
