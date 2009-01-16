# STDP
# See BEP-2-STDP
from inspection import *
from equations import *
import re

class STDP(object):
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
        if eqs_obj._eq_names!=[] or eqs_obj._alias!=[]:
            raise Exception,"There should be only differential equations"
        
        # Check units
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
        
        # create virtual groups (inherit NeuronGroup; Group?), pre and post
        
        # Pre code
        for var in vars_pre: # presynaptic variables (vectorisation)
            pre=re.sub(r'\b'+var+r'\b',var+'[spikes]',pre)
        pre=re.sub(r'\bw\b','w[spikes,:]') # synaptic weight
        # Post code
        for var in vars_post: # presynaptic variables (vectorisation)
            post=re.sub(r'\b'+var+r'\b',var+'[spikes]',post)
        post=re.sub(r'\bw\b','w[:,spikes]') # synaptic weight
        
        # bounds: add one line to pre/post code (clip(w,min,max,w))
        if bounds is not None: # would that work with SparseVector? probably not...
            min,max=bounds
            pre+='\nclip(w[spikes,:],%(min),%(max),w[spikes,:])' % {'min':min,'max':max}
            post+='\nclip(w[:,spikes],%(min),%(max),w[:,spikes])' % {'min':min,'max':max}
        
        # event-driven code; do some speed tests
        # create forward and backward Connection objects; propagate does pre or post code and
        #   event-driven updates
    