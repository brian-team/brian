# STDP
# See BEP-2-STDP
from inspection import *
from equations import *
import re

class STDP(object): # NetworkOperation?
    '''
    Spike-timing-dependent plasticity
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
        # Disallow static equations and aliases (for now)
        if eqs_obj._eq_names!=[] or eqs_obj._eq_names!=[]:
            print eqs_obj._eq_names,eqs_obj._eq_names
            raise Exception,"There should be only differential equations"
        
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
        
        # Additional check: modification of presynaptic variables should not depend on postsynaptic
        #   variables
        
        # One difficulty here: we need to determine the dependency of differential variables
        # (a variable may not be directly modified by spikes but depend on a modified var)
        
        # check pre/post consistency (TODO: see static variables)
        for var in vars_pre:
            for x in get_identifiers(eqs_obj._string[var]):
                if x in vars_post:
                    raise Exception,"Presynaptic variable "+var+" depends on postsynaptic variable "+x
        for var in vars_post:
            for x in get_identifiers(eqs_obj._string[var]):
                if x in vars_pre:
                    raise Exception,"Postsynaptic variable "+var+" depends on presynaptic variable "+x
        
        # separate differential equations in pre/post
        eqs_pre=Equations()
        eqs_post=Equations()
        for line in eqs.splitlines():
            eq=Equations(line,level=level+1)
            if eq._diffeq_names==[]: # no differential equation
                pass # empty line
            elif eq._diffeq_names[0] in vars_pre:
                eqs_pre+=eq
            elif eq._diffeq_names[0] in vars_post:
                eqs_post+=eq
            else: # what do we do?
                pass
        
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
            min,max=bounds
            pre+='\nclip(w[spikes,:],%(min)f,%(max)f,w[spikes,:])' % {'min':min,'max':max}
            post+='\nclip(w[:,spikes],%(min)f,%(max)f,w[:,spikes])' % {'min':min,'max':max}
        
        # Create namespace
        # Compile code
        
        # create virtual groups (inherit NeuronGroup; Group?), pre and post        
        # event-driven code; do some speed tests
        # create forward and backward Connection objects; propagate does pre or post code and
        #   event-driven updates
    
if __name__=='__main__':
    from brian import *
    P=NeuronGroup(10,model='dv/dt=-v/(10*ms):1')
    C=Connection(P,P,'v')
    tau_pre=20*ms
    tau_post=20*ms
    A_pre=.1
    A_post=-.2
    eqs_stdp='''
    dA_pre/dt=-A_pre/tau_pre : 1
    dA_post/dt=-A_post/tau_post : 1
    '''
    stdp=STDP(C,eqs=eqs_stdp,pre='A_pre+=dA_pre;w+=A_post',
              post='A_post+=dA_post;w+=A_pre',bounds=(0,1))
    