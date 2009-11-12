'''
Spike-timing-dependent plasticity
'''
# See BEP-2-STDP
from inspection import *
from equations import *
from monitor import SpikeMonitor, RecentStateMonitor
from network import NetworkOperation
from neurongroup import NeuronGroup
from stateupdater import get_linear_equations,LinearStateUpdater
from scipy.linalg import expm
from scipy import dot,eye,zeros,array,clip,exp,Inf
from stdunits import ms
from connection import DelayConnection, DenseConstructionMatrix, SparseConnectionVector
import re
from utils.documentation import flattened_docstring
from copy import copy
import warnings
from itertools import izip
from numpy import arange, floor
from clock import Clock
from units import second

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
        if len(spikes):
            self._namespace['spikes']=spikes
            self._namespace['w']=self.C.W
            exec self._code in self._namespace

class DelayedSTDPUpdater(SpikeMonitor):

    def __init__(self, C, reverse, delay_expr, max_delay,
                 vars, other_vars, varmon, othervarmon, code, namespace, delay=0*ms):
        if reverse:
            source, target = C.target, C.source
            self.get_times_seq = 'get_cols'
        else:
            source, target = C.source, C.target
            self.get_times_seq = 'get_rows'
        super(DelayedSTDPUpdater, self).__init__(source, record=False, delay=delay)
        self._code = code # update code
        self._namespace = namespace # code namespace
        self.C = C
        self.vars = vars
        self.other_vars = other_vars
        self.varmon = varmon
        self.othervarmon = othervarmon
        #self._arange = arange(len(source))
        delay_expr = re.sub(r'\bmax_delay\b', str(float(max_delay)), delay_expr)
        delay_expr = 'lambda d:'+delay_expr
#        print 'Delay expr:', delay_expr
        self.delay_expr = eval(delay_expr)
        
    def propagate(self, spikes):
        if len(spikes):
            if isinstance(self.get_times_seq, str):
#                self.wget = getattr(self.C.W, self.get_times_seq)
                self.get_times_seq = getattr(self.C.delayvec, self.get_times_seq)
            times_seq = self.get_times_seq(spikes)
            times_seq = [self.delay_expr(times) for times in times_seq]
            for var in self.other_vars:
                delayed_values = self.othervarmon[var].get_past_values_sequence(times_seq)
#                print var, times_seq, '\n', delayed_values
#                for dv, w in zip(delayed_values, self.wget(spikes)):
#                    print dv.ind, w.ind, dv.ind is w.ind
                self._namespace[var+'__delayed_values_seq'] = delayed_values
            self._namespace['spikes'] = spikes
            self._namespace['w'] = self.C.W
            exec self._code in self._namespace

class STDP(NetworkOperation):
    '''
    Spike-timing-dependent plasticity    

    Initialised with arguments:

    ``C``
        Connection object to apply STDP to.
    ``eqs``
        Differential equations (with units)
    ``pre``
        Python code for presynaptic spikes, use the reserved symbol ``w`` to
        refer to the synaptic weight.
    ``post``
        Python code for postsynaptic spikes, use the reserved symbol ``w`` to
        refer to the synaptic weight.
    ``wmax``
        Maximum weight (default unlimited), weights are restricted to be within
        0 and this value.
    ``delay_pre``
        Presynaptic delay
    ``delay_post``
        Postsynaptic delay (backward propagating spike)
    
    The STDP object works by specifying a set of differential equations
    associated to each synapse (``eqs``) and two rules to specify what should
    happen when a presynaptic neuron fires (``pre``) and when a postsynaptic
    neuron fires (``post``). The equations should be standard set of equations
    in the usual string format. The ``pre`` and ``post`` rules should be a
    sequence of statements to be executed triggered on pre- and post-synaptic
    spikes. The sequence of statements can be separated by a ``;`` or by
    using a multiline string. The reserved symbol ``w`` can be used to refer
    to the synaptic weight of the associated synapse.
    
    This framework allows you to implement most STDP rules. Specifying
    differential equations and pre- and post-synaptic event code allows for a
    much more efficient implementation than specifying, for example, the
    spike pair weight modification function, but does unfortunately require
    transforming the definition into this form.
    
    There is one restriction on the equations that can be implemented in this
    system, they need to be separable into independent pre- and post-synaptic
    systems (this is done automatically). In this way, synaptic variables and
    updates can be stored per neuron rather than per synapse. 
    
    **Example**
    
    ::
    
        eqs_stdp = """
        dA_pre/dt  = -A_pre/tau_pre   : 1
        dA_post/dt = -A_post/tau_post : 1
        """
        stdp = STDP(synapses, eqs=eqs_stdp, pre='A_pre+=delta_A_pre; w+=A_post',
                    post='A_post+=delta_A_post; w+=A_pre', wmax=gmax)
    
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
            #raise AttributeError,"STDP does not handle heterogeneous connections yet."
            warnings.warn("STDP with heterogeneous delays is experimental!")
#            if not isinstance(C.W, DenseConstructionMatrix):
#                raise TypeError('STDP only works with dense DelayConnection objects at the moment.')
        NetworkOperation.__init__(self,lambda:None,clock=clock)
        # Merge multi-line statements
        #eqs=re.sub('\\\s*?\n',' ',eqs)
        # Convert to equations object
        if isinstance(eqs,Equations):
            eqs_obj=eqs
        else:
            eqs_obj=Equations(eqs,level=level+1)
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

        if isinstance(C, DelayConnection):
            G_pre_monitors = {} # these get values put in them later
            G_post_monitors = {}
            max_delay = C._max_delay*C.target.clock.dt            

            def gencode(incode, vars, other_vars, wreplacement):
#                wreplacement = re.sub(r':', 'neuron_indices[_j]', wreplacement)
                incode_lines = [line.strip() for line in incode.split('\n')]
                outcode_immediate = 'for _i in spikes:\n'
                # delayed variables
                outcode_delayed = 'for _j, _i in enumerate(spikes):\n'
                for var in other_vars:
                    outcode_delayed += '    '+var+'__delayed = '+var+'__delayed_values_seq[_j]\n'                
                for line in incode_lines:
                    m = re.search(r'\bw\b\s*[^><=]?=', line) # lines of the form w = ..., w *= ..., etc.
                    for var in vars:
                        line = re.sub(r'\b'+var+r'\b', var+'[_i]', line)
                    for var in other_vars:
                        line = re.sub(r'\b'+var+r'\b', var+'__delayed', line)
                    if m:
                        outcode_delayed += '    '+line+'\n'
                    else:
                        outcode_immediate += '    '+line+'\n'
                outcode_delayed = re.sub(r'\bw\b', wreplacement, outcode_delayed)
                outcode_delayed += '\n    %(w)s = clip(%(w)s, 0, %(max)f)' % {'max':wmax, 'w':wreplacement}
                return (outcode_immediate, outcode_delayed)

            
#            def gencode(incode, vars, other_vars, wreplacement, varlen, delay_expr, dvec_get):
#                incode_lines = [line.strip() for line in incode.split('\n')]
#                outcode_immediate = 'for i in spikes:\n'
#                # delayed variables
#                outcode_delayed = 'jinds = arange('+str(varlen)+', dtype=int)\n'
#                outcode_delayed += 'varmon0 = var_monitors.itervalues().next()\n'
#                outcode_delayed += 'cur_delay_ind = varmon0.current_time_index\n'
#                outcode_delayed += 'invtargetdt = 1.0/varmon0.clock._dt\n'
#                outcode_delayed += 'max_delay = varmon0.num_duration\n'
#                outcode_delayed += 'dvecrows = delayconn.delayvec.'+dvec_get+'(spikes)\n'
#                outcode_delayed += 'for dvecrow, i in izip(dvecrows, spikes):\n'
#                # TODO: this only works for structure='dense'
#                delay_expr = re.sub(r'\bd\b', 'dvecrow', delay_expr)
#                delay_expr = re.sub(r'\bmax_delay\b', str(float(max_delay)), delay_expr)
#                outcode_delayed += '    inds = (cur_delay_ind-1+array(floor(invtargetdt*('+delay_expr+')), dtype=int))%max_delay\n'
#                for var in other_vars:
#                    outcode_delayed += '    '+var+'__delayed = var_monitors["'+var+'"]._values[inds, jinds]\n'
#                for line in incode_lines:
#                    m = re.search(r'\bw\b\s*[^><=]?=', line) # lines of the form w = ..., w *= ..., etc.
#                    for var in vars:
#                        line = re.sub(r'\b'+var+r'\b', var+'[i]', line)
#                    for var in other_vars:
#                        line = re.sub(r'\b'+var+r'\b', var+'__delayed', line)
#                    if m:
#                        outcode_delayed += '    '+line+'\n'
#                    else:
#                        outcode_immediate += '    '+line+'\n'
#                outcode_delayed = re.sub(r'\bw\b', wreplacement, outcode_delayed)
#                outcode_delayed += '\n    %(w)s = clip(%(w)s, 0, %(max)f)' % {'max':wmax, 'w':wreplacement}
##                print 'Delayed:'
##                print outcode_delayed
##                print 'Immediate:'
##                print outcode_immediate
#                return (outcode_immediate, outcode_delayed)
            pre_immediate, pre_delayed = gencode(pre, vars_pre, vars_post, 'w[_i,:]')
            post_immediate, post_delayed = gencode(post, vars_post, vars_pre, 'w[:,_i]')
            pre_delay_expr = 'max_delay-d'
            post_delay_expr = 'd'
            pre_namespace['enumerate'] = enumerate
            pre_namespace['clip'] = clip
            post_namespace['enumerate'] = enumerate
            post_namespace['clip'] = clip
#            print 'pre_immediate:'
#            print pre_immediate
#            print 'pre_delayed:'
#            print pre_delayed
#            print 'post:'
#            print post_immediate+post_delayed
            pre_code_immediate = compile(pre_immediate, "Presynaptic code immediate", "exec")
            pre_code_delayed = compile(pre_delayed, "Presynaptic code delayed", "exec")
            post_code = compile(post_immediate+post_delayed, "Postsynaptic code", "exec")
        else:
            # Indent and loop
            pre=re.compile('^',re.M).sub('    ',pre)
            post=re.compile('^',re.M).sub('    ',post)
            pre='for _i in spikes:\n'+pre
            post='for _i in spikes:\n'+post
    
            # Pre code
            for var in vars_pre: # presynaptic variables (vectorisation)
                pre=re.sub(r'\b'+var+r'\b',var+'[_i]',pre)
            pre=re.sub(r'\bw\b','w[_i,:]',pre) # synaptic weight
            # Post code
            for var in vars_post: # postsynaptic variables (vectorisation)
                post=re.sub(r'\b'+var+r'\b',var+'[_i]',post)
            post=re.sub(r'\bw\b','w[:,_i]',post) # synaptic weight
            
            # Bounds: add one line to pre/post code (clip(w,min,max,w))
            # or actual code? (rather than compiled string)
            pre+='\n    w[_i,:]=clip(w[_i,:],0,%(max)f)' % {'max':wmax}
            post+='\n    w[:,_i]=clip(w[:,_i],0,%(max)f)' % {'max':wmax}
            pre_namespace['clip']=clip
            post_namespace['clip']=clip
            
            # Compile code
            pre_code=compile(pre,"Presynaptic code","exec")
            post_code=compile(post,"Postsynaptic code","exec")
        
        # Report
#        print pre_namespace
#        print pre
#        print
#        print post_namespace
#        print post
#        print
#        print M_pre
#        print M_post
        
        # Delays
        if isinstance(C, DelayConnection):
            if delay_pre is not None or delay_post is not None:
                raise ValueError("Must use delay_pre=delay_post=None for the moment.")
            max_delay = C._max_delay*C.target.clock.dt
            # Ensure that the source and target neuron spikes are kept for at least the
            # DelayConnection's maximum delay
            C.source.set_max_delay(max_delay)
            C.target.set_max_delay(max_delay)
            # create forward and backward Connection objects or SpikeMonitor objects
            pre_updater_immediate = STDPUpdater(C.source, C, vars=vars_pre,
                                           code=pre_code_immediate, namespace=pre_namespace, delay=0*ms)
            pre_updater_delayed = DelayedSTDPUpdater(C, reverse=False, delay_expr=pre_delay_expr, max_delay=max_delay,
                                            vars=vars_pre, other_vars=vars_post,
                                            varmon=G_pre_monitors, othervarmon=G_post_monitors,
                                            code=pre_code_delayed, namespace=pre_namespace, delay=max_delay)
            post_updater = DelayedSTDPUpdater(C, reverse=True, delay_expr=post_delay_expr, max_delay=max_delay,
                                            vars=vars_post, other_vars=vars_pre,
                                            varmon=G_post_monitors, othervarmon=G_pre_monitors,
                                            code=post_code, namespace=post_namespace, delay=0*ms)
#            pre_updater_immediate = STDPUpdater(C.source, C, vars=vars_pre,
#                                           code=pre_code_immediate, namespace=pre_namespace, delay=0*ms)
#            pre_updater_delayed = STDPUpdater(C.source, C, vars=vars_pre,
#                                           code=pre_code_delayed, namespace=pre_namespace, delay=max_delay)
#            post_updater = STDPUpdater(C.target, C, vars=vars_post,
#                                       code=post_code, namespace=post_namespace, delay=0*ms)
            updaters = [pre_updater_immediate, pre_updater_delayed, post_updater]
            self.contained_objects += updaters
        else:
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
            updaters = [pre_updater, post_updater]
            self.contained_objects += [pre_updater, post_updater]
        
        # Neuron groups
        G_pre=NeuronGroup(len(C.source),model=LinearStateUpdater(M_pre,clock=self.clock),clock=self.clock)
        G_post=NeuronGroup(len(C.target),model=LinearStateUpdater(M_post,clock=self.clock),clock=self.clock)
        G_pre._S[:]=0
        G_post._S[:]=0
        
        # Put variables in namespaces
        vars_pre_ind = {}
        for i, var in enumerate(vars_pre):
            vars_pre_ind[var] = i
            for updater in updaters:
                updater._namespace[var] = G_pre._S[i]

        vars_post_ind = {}
        for i, var in enumerate(vars_post):
            vars_post_ind[var] = i
            for updater in updaters:
                updater._namespace[var] = G_post._S[i]

        if isinstance(C, DelayConnection):
            self.G_pre_monitors = G_pre_monitors
            self.G_post_monitors = G_post_monitors
            self.G_pre_monitors.update(((var, RecentStateMonitor(G_pre, vars_pre_ind[var], duration=(C._max_delay+1)*C.target.clock.dt, clock=G_pre.clock)) for var in vars_pre))
            self.G_post_monitors.update(((var, RecentStateMonitor(G_post, vars_post_ind[var], duration=(C._max_delay+1)*C.target.clock.dt, clock=G_post.clock)) for var in vars_post))
            self.contained_objects += self.G_pre_monitors.values()
            self.contained_objects += self.G_post_monitors.values()
        
        self.contained_objects += [G_pre, G_post]
    
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
                 update='additive',delay_pre=None,delay_post=None,clock=None):
        if wmax is None:
            raise AttributeError,"You must specify the maximum synaptic weight"
        wmax=float(wmax) # removes units

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
        STDP.__init__(self,C,eqs=eqs,pre=pre,post=post,wmax=wmax,delay_pre=delay_pre,delay_post=delay_post,clock=clock)

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
