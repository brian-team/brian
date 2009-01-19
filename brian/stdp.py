# STDP
# See BEP-2-STDP
from inspection import *
from equations import *
from monitor import SpikeMonitor
from network import NetworkOperation
from stateupdater import get_linear_equations
from scipy.linalg import expm
from scipy import dot,eye,zeros,array
import re

__all__=['STDPUpdater','STDP','dependency_matrix']

class STDPUpdater(SpikeMonitor):
    '''
    Updates STDP variables at spike times
    '''
    def __init__(self,source,vars,M,code,namespace,delay=0):
        '''
        source = source group
        vars = variable names
        M = matrix of the linear differential system
        code = code to execute for every spike
        namespace = namespace for the code
        delay = transmission delay 
        '''
        super(STDPUpdater,self).__init__(source,record=False,delay=delay)
        self._code=code # update code
        self._namespace=namespace # code namespace
        # State variables
        self.S=zeros((len(vars),len(source)))
        for i,var in enumerate(vars):
            self._namespace[var]=self.S[i]
        self.lastt=zeros(len(vars)) # time since last update
        self.M=M
        self.clock=source.clock
        
    def propagate(self,spikes):
        self._namespace['spikes']=spikes
        t=self.clock.t
        for i in spikes:
            self.S[:,i]=expm(self.M*(t-self.lastt[i]))*self.S[:,i]
        self.lastt[spikes]=t
        exec self._code in self._namespace

class STDP(NetworkOperation):
    '''
    Spike-timing-dependent plasticity
    
    TODO: set pre and post transmission delays (should sum to the connection delay)
    '''
    def __init__(self,C,eqs,pre,post,bounds=None,level=0):
        '''
        C: connection object
        eqs: differential equations (with units)
        pre: Python code for presynaptic spikes
        post: Python code for postsynaptic spikes
        bounds: bounds on the weights, e.g. (0,gmax) (default no bounds)
        '''
        # Merge multi-line statements
        eqs=re.sub('\\\s*?\n',' ',eqs)
        # Convert to equations object
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
        pre_namespace['w']=C.W
        post_namespace['w']=C.W

        # Pre code
        for var in vars_pre: # presynaptic variables (vectorisation)
            pre=re.sub(r'\b'+var+r'\b',var+'[spikes]',pre)
        pre=re.sub(r'\bw\b','w[spikes,:]',pre) # synaptic weight
        # Post code
        for var in vars_post: # presynaptic variables (vectorisation)
            post=re.sub(r'\b'+var+r'\b',var+'[spikes]',post)
        post=re.sub(r'\bw\b','w[:,spikes]',post) # synaptic weight
        
        # Bounds: add one line to pre/post code (clip(w,min,max,w))
        if bounds is not None: # would that work with SparseVector? probably not...
            # or actual code? (rather than compiled string)
            min,max=bounds
            pre+='\nclip(w[spikes,:],%(min)f,%(max)f,w[spikes,:])' % {'min':min,'max':max}
            post+='\nclip(w[:,spikes],%(min)f,%(max)f,w[:,spikes])' % {'min':min,'max':max}
                
        # Compile code
        pre_code=compile(pre,"Presynaptic code","exec")
        post_code=compile(pre,"Postsynaptic code","exec")
        
        # create virtual groups (inherit NeuronGroup; Group?), pre and post
        
        # Add update code to pre and post
        # 1-dimensional case (exponential STDP)
        #if (len(vars_pre)==1) and (len(vars_post)==1):
        #    vars_pre[0]+'[spikes]*=exp(%(a)f*t[spikes])' % Mpre[0]
        #    vars_post[0]+'[spikes]*=exp(%(a)f*t[spikes])' % Mpost[0]
        
        # create forward and backward Connection objects or SpikeMonitor objects
        pre_updater=STDPUpdater(C.source,vars=vars_pre,M=M_pre,code=pre_code,namespace=pre_namespace)
        post_updater=STDPUpdater(C.source,vars=vars_post,M=M_post,code=post_code,namespace=post_namespace)
        
    def __call__(self):
        pass
    
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
    from brian import *
    from time import time
    P=NeuronGroup(10,model='dv/dt=-v/(10*ms):1')
    C=Connection(P,P,'v')
    tau_pre=20*ms
    tau_post=20*ms
    dA_pre=.1
    dA_post=-.2
    eqs_stdp='''
    dA_pre/dt=(x-A_pre)/tau_pre : 1
    dx/dt=(y-x)/tau_pre : 1
    dy/dt=-y/tau_pre : 1
    dA_post/dt=-A_post/tau_post : 1
    '''
    t1=time()
    stdp=STDP(C,eqs=eqs_stdp,pre='A_pre+=dA_pre;w+=A_post',
              post='A_post+=dA_post;w+=A_pre',bounds=(0,1))
    print time()-t1