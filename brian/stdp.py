'''
Spike-timing-dependent plasticity
'''
# See BEP-2-STDP
from inspection import *
from equations import *
from monitor import SpikeMonitor
from network import NetworkOperation
from neurongroup import NeuronGroup
from stateupdater import get_linear_equations,LinearStateUpdater
from scipy.linalg import expm
from scipy import dot,eye,zeros,array,clip,exp,Inf
from stdunits import ms
from connection import DelayConnection
import re

__all__=['STDP','ExponentialSTDP']

class STDPUpdater(SpikeMonitor):
    '''
    Updates STDP variables at spike times
    '''
    def __init__(self,source,C,vars,code,namespace,delay=0*ms):
        '''
        source = source group
        C = connection
        vars = variable names
        M = matrix of the linear differential system
        code = code to execute for every spike
        namespace = namespace for the code
        delay = transmission delay 
        '''
        super(STDPUpdater,self).__init__(source,record=False,delay=delay)
        self._code=code # update code
        self._namespace=namespace # code namespace
        self.C=C
        
    def propagate(self,spikes):
        self._namespace['spikes']=spikes
        self._namespace['w']=self.C.W
        exec self._code in self._namespace

class STDP(NetworkOperation):
    '''
    Spike-timing-dependent plasticity    

    Initialised with arguments:

    ``C``
        connection object
    ``eqs``
        differential equations (with units)
    ``pre``
        Python code for presynaptic spikes
    ``post``
        Python code for postsynaptic spikes
    ``wmax``
        maximum weight (default unlimited)
    ``delay_pre``
        presynaptic delay
    ``delay_post``
        postsynaptic delay (backward propagating spike)
    
    **Example**
    
    ::
    
        eqs_stdp = """
        dA_pre/dt  = -A_pre/tau_pre   : 1
        dA_post/dt = -A_post/tau_post : 1
        """
        stdp = STDP(synapses, eqs=eqs_stdp, pre='A_pre+=dA_pre; w+=A_post',
                    post='A_post+=dA_post; w+=A_pre', wmax=gmax)
    
    **Technical details**
    
    The equations are split into two groups, pre and post. Two groups are created
    to carry these variables and to update them (these are implemented as
    :class:`NeuronGroup` objects). As well as propagating spikes from the source
    and target of ``C`` via ``C``, spikes are also propagated to the respective
    groups created. At spike propagation time the weight values are updated.
    '''
    #TODO: use equations instead of linearupdater (-> nonlinear equations possible)
    #TODO: allow pre and postsynaptic group variables
    def __init__(self,C,eqs,pre,post,wmax=Inf,level=0,clock=None,delay_pre=None,delay_post=None):
        '''
        C: connection object
        eqs: differential equations (with units)
        pre: Python code for presynaptic spikes
        post: Python code for postsynaptic spikes
        wmax: maximum weight (default unlimited)
        delay_pre: presynaptic delay
        delay_post: postsynaptic delay (backward propagating spike)
        '''
        if isinstance(C,DelayConnection):
            raise AttributeError,"STDP does not handle heterogeneous connections yet."
        NetworkOperation.__init__(self,lambda:None,clock=clock)
        # Merge multi-line statements
        #eqs=re.sub('\\\s*?\n',' ',eqs)
        # Convert to equations object
        if isinstance(eqs,Equations):
            eqs_obj=eqs
        else:
            eqs_obj=Equations(eqs,level=level+1)
        
        # Check units
        eqs_obj.compile_functions()
        eqs_obj.check_units()
        # Check that equations are linear
        if not eqs_obj.is_linear():
            raise Exception,"Only linear differential equations are handled"
        # Get variable names
        vars=eqs_obj._diffeq_names
        # Find which ones are directly modified (e.g. regular expression matching; careful with comments)
        vars_pre=[var for var in vars if var in modified_variables(pre)]
        vars_post=[var for var in vars if var in modified_variables(post)]
        
        # Additional check TODO: modification of presynaptic variables should not depend on postsynaptic
        #   variables
        
        # Get the matrix of the differential system
        M,_=get_linear_equations(eqs_obj)
        D=dependency_matrix(M)
        
        # Collect dependent variables
        dependent_pre=zeros(M.shape[0])
        dependent_post=zeros(M.shape[0])
        for i,var in enumerate(vars):
            if var in vars_pre:
                dependent_pre+=D[i,:]
            elif var in vars_post:
                dependent_post+=D[i,:]
        index_pre=(dependent_pre>0).nonzero()[0]
        index_post=(dependent_post>0).nonzero()[0]
        vars_pre=array(vars)[index_pre]
        vars_post=array(vars)[index_post]
        
        # Check pre/post consistency
        shared_vars=set(vars_pre).intersection(vars_post)
        if shared_vars!=set([]):
            raise Exception,str(list(shared_vars))+" are both presynaptic and postsynaptic!"
        
        # Split the matrix M
        M_pre=M[index_pre,:][:,index_pre]
        M_post=M[index_post,:][:,index_post]
        
        # Create namespaces for pre and post codes
        pre_namespace=namespace(pre,level=level+1)
        post_namespace=namespace(post,level=level+1)

        # Indent and loop
        pre=re.compile('^',re.M).sub('    ',pre)
        post=re.compile('^',re.M).sub('    ',post)
        pre='for i in spikes:\n'+pre
        post='for i in spikes:\n'+post

        # Pre code
        for var in vars_pre: # presynaptic variables (vectorisation)
            pre=re.sub(r'\b'+var+r'\b',var+'[i]',pre)
        pre=re.sub(r'\bw\b','w[i,:]',pre) # synaptic weight
        # Post code
        for var in vars_post: # postsynaptic variables (vectorisation)
            post=re.sub(r'\b'+var+r'\b',var+'[i]',post)
        post=re.sub(r'\bw\b','w[:,i]',post) # synaptic weight
        
        # Bounds: add one line to pre/post code (clip(w,min,max,w))
        # or actual code? (rather than compiled string)
        pre+='\n    w[i,:]=clip(w[i,:],0,%(max)f)' % {'max':wmax}
        post+='\n    w[:,i]=clip(w[:,i],0,%(max)f)' % {'max':wmax}
        pre_namespace['clip']=clip
        post_namespace['clip']=clip
        
        # Compile code
        pre_code=compile(pre,"Presynaptic code","exec")
        post_code=compile(post,"Postsynaptic code","exec")
        
        # Delays
        connection_delay=C.delay*C.source.clock.dt
        if (delay_pre is None) and (delay_post is None): # same delays as the Connnection C
            delay_pre=connection_delay
            delay_post=0*ms
        elif delay_pre is None:
            delay_pre=connection_delay-delay_post
            if delay_pre<0*ms: raise AttributeError,"Postsynaptic delay is too large"
        elif delay_post is None:
            delay_post=connection_delay-delay_pre
            if delay_post<0*ms: raise AttributeError,"Postsynaptic delay is too large"
        
        # create forward and backward Connection objects or SpikeMonitor objects
        pre_updater=STDPUpdater(C.source,C,vars=vars_pre,code=pre_code,namespace=pre_namespace,delay=delay_pre)
        post_updater=STDPUpdater(C.target,C,vars=vars_post,code=post_code,namespace=post_namespace,delay=delay_post)
        
        # Neuron groups
        G_pre=NeuronGroup(len(C.source),model=LinearStateUpdater(M_pre,clock=self.clock))
        G_post=NeuronGroup(len(C.target),model=LinearStateUpdater(M_post,clock=self.clock))
        G_pre._S[:]=0
        G_post._S[:]=0
        
        # Put variables in namespaces
        for i,var in enumerate(vars_pre):
            pre_updater._namespace[var]=G_pre._S[i]
            post_updater._namespace[var]=G_pre._S[i]

        for i,var in enumerate(vars_post):
            pre_updater._namespace[var]=G_post._S[i]
            post_updater._namespace[var]=G_post._S[i]
        
        self.contained_objects=[pre_updater,post_updater,G_pre,G_post]
    
    def __call__(self):
        pass

class ExponentialSTDP(STDP):
    '''
    Exponential STDP.
    
    Initialised with the following arguments:
    
    ``taup``, ``taum``, ``Ap``, ``Am``
        Synaptic weight change (relative to the maximum weight wmax)::
        
            f(s) = Ap*exp(-s/taup) if s >0
            f(s) = Am*exp(s/taum) if s <0
                    
    ``interactions``
      * 'all': contributions from all pre-post pairs are added
      * 'nearest': only nearest-neighbour pairs are considered
      * 'nearest_pre': nearest presynaptic spike, all postsynaptic spikes
      * 'nearest_post': nearest postsynaptic spike, all presynaptic spikes
          
    ``wmax``
        maximum synaptic weight
        
    ``update``
      * 'additive': modifications are additive (independent of synaptic weight)
        (or "hard bounds")
      * 'multiplicative': modifications are multiplicative (proportional to w)
        (or "soft bounds")
      * 'mixed': depression is multiplicative, potentiation is additive
    
    See documentation for :class:`STDP` for more details.
    '''
    def __init__(self,C,taup,taum,Ap,Am,interactions='all',wmax=None,
                 update='additive',delay_pre=None,delay_post=None):
        if wmax is None:
            raise AttributeError,"You must specify the maximum synaptic weight"

        eqs=Equations('''
        dA_pre/dt=-A_pre/taup : 1
        dA_post/dt=-A_post/taum : 1''',taup=taup,taum=taum,wmax=wmax)
        if interactions=='all':
            pre='A_pre+=Ap'
            post='A_post+=Am'
        elif interactions=='nearest':
            pre='A_pre=Ap'
            post='A_post=Am'
        elif interactions=='nearest_pre':
            pre='A_pre=Ap'
            post='A_post=+Am'
        elif interactions=='nearest_post':
            pre='A_pre+=Ap'
            post='A_post=Am'
        else:
            raise AttributeError,"Unknown interaction type "+interactions
        
        if update=='additive':
            Ap*=wmax
            Am*=wmax
            pre+='\nw+=A_post'
            post+='\nw+=A_pre'
        elif update=='multiplicative':
            if Am<0:
                pre+='\nw*=(1+A_post)'
            else:
                pre+='\nw+=(wmax-w)*A_post'
            if Ap<0:
                post+='\nw*=(1+A_pre)'
            else:
                post+='\nw+=(wmax-w)*A_pre'
        elif update=='mixed':
            if Am<0 and Ap>0:
                Ap*=wmax
                pre+='\nw*=(1+A_post)'
                post+='\nw+=A_pre'
            elif Am>0 and Ap<0:
                Am*=wmax
                post+='\nw*=(1+A_pre)'
                pre+='\nw+=A_post'
            else:
                if Am>0:
                    raise AttributeError,"There is no depression in STDP rule"
                else:
                    raise AttributeError,"There is no potentiation in STDP rule"
        else:
            raise AttributeError,"Unknown update type "+update
        STDP.__init__(self,C,eqs=eqs,pre=pre,post=post,wmax=wmax,delay_pre=delay_pre,delay_post=delay_post)

# TODO: insert it in Equations, as a method returning an Equations object
# for a given list of variables, with dependent variables
def dependency_matrix(A):
    '''
    A is the (square) matrix of a differential system (or a difference system).
    Returns a matrix (Mij) where Mij==True iff variable i depends on variable j.
    '''
    n=A.shape[0] # check if square?
    D=A!=0
    U=eye(n)
    M=eye(n)
    for _ in range(n-1):
        U=dot(M,D)
        M+=U
    return M!=0

if __name__=='__main__':
    pass
