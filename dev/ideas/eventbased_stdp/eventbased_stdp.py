"""
Event-based STDP
"""
from brian import *
from brian.utils.documentation import flattened_docstring
from brian.inspection import *
from brian.equations import *
from brian.optimiser import freeze
import re

class EventBasedSTDP(NetworkOperation):
    '''
    Spike-timing-dependent plasticity
    Event-based implementation
    
    **Example**
    
    ::
    
        eqs_stdp = """
        A_pre   : 1
        A_post  : 1
        """
        pre = """
        A_pre+=delta_A_pre+A_pre*exp(-(t-t_pre)/tau_pre)
        w+=A_post
        """
        post = """
        A_post+=delta_A_post+A_post*exp(-(t-t_post)/tau_post)
        w+=A_pre
        """
        stdp = EventBasedSTDP(synapses, eqs=eqs_stdp,pre=pre,post=post, wmax=gmax)
                    
    '''
    def __init__(self, C, eqs, pre, post, wmin=0, wmax=Inf, level=0, clock=None, delay_pre=None, delay_post=None):
        '''
        C: connection object
        eqs: equations (with units)
        pre: Python code for presynaptic spikes
        post: Python code for postsynaptic spikes
        wmax: maximum weight (default unlimited)
        delay_pre: presynaptic delay
        delay_post: postsynaptic delay (backward propagating spike)
        '''
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        # Convert to equations object
        if isinstance(eqs, Equations):
            eqs_obj = eqs
        else:
            eqs_obj = Equations(eqs, level=level + 1)
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
        # Get variable names
        vars = eqs_obj._diffeq_names

        # Create namespaces for pre and post codes
        pre_namespace = namespace(pre, level=level + 1)
        post_namespace = namespace(post, level=level + 1)
        pre_namespace['clip'] = clip
        post_namespace['clip'] = clip

        # freeze pre and post (otherwise units will cause problems)
        all_vars = list(vars) + ['w','t','t_pre','t_post']
        pre = '\n'.join(freeze(line.strip(), all_vars, pre_namespace) for line in pre.split('\n'))
        post = '\n'.join(freeze(line.strip(), all_vars, post_namespace) for line in post.split('\n'))

        # Create synaptic variables
        self.var=dict()
        for x in vars:
            self.var[x]=C.W.connection_matrix()
            self.var[x][:]=0 # reset values
        
        # Create code
        # Indent and loop
        pre = re.compile('^', re.M).sub('    ', pre)
        post = re.compile('^', re.M).sub('    ', post)
        pre = 'for _i in spikes:\n' + pre
        post = 'for _i in spikes:\n' + post

        # Pre/post code
        for var in vars+['w']: # presynaptic variables (vectorisation)
            pre = re.sub(r'\b' + var + r'\b', var + '[_i,:]', pre)
            post = re.sub(r'\b' + var + r'\b', var + '[:,_i]', post)

        # Bounds: add one line to pre/post code (clip(w,min,max,w))
        # or actual code? (rather than compiled string)
        pre += '\n    w[_i,:]=clip(w[_i,:],%(min)e,%(max)e)' % {'min':wmin, 'max':wmax}
        post += '\n    w[:,_i]=clip(w[:,_i],%(min)e,%(max)e)' % {'min':wmin, 'max':wmax}
        log_debug('brian.stdp', 'PRE CODE:\n'+pre)
        log_debug('brian.stdp', 'POST CODE:\n'+post)
        # Compile code
        pre_code = compile(pre, "Presynaptic code", "exec")
        post_code = compile(post, "Postsynaptic code", "exec")

        # Delays
        connection_delay = C.delay * C.source.clock.dt
        if (delay_pre is None) and (delay_post is None): # same delays as the Connnection C
            delay_pre = connection_delay
            delay_post = 0 * ms
        elif delay_pre is None:
            delay_pre = connection_delay - delay_post
            if delay_pre < 0 * ms: raise AttributeError, "Postsynaptic delay is too large"
        elif delay_post is None:
            delay_post = connection_delay - delay_pre
            if delay_post < 0 * ms: raise AttributeError, "Postsynaptic delay is too large"
        
        # create forward and backward Connection objects or SpikeMonitor objects
        #pre_updater = STDPUpdater(C.source, C, vars=vars_pre, code=pre_code, namespace=pre_namespace, delay=delay_pre)
        #post_updater = STDPUpdater(C.target, C, vars=vars_post, code=post_code, namespace=post_namespace, delay=delay_post)
        #updaters = [pre_updater, post_updater]
        #self.contained_objects += [pre_updater, post_updater]

if __name__ == '__main__':
    pass
