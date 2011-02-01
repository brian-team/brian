from numpy import *
if __name__ == '__main__':
    from brian import *
    from brian.experimental.new_c_propagate import *
    from brian.experimental.codegen.expressions import *
    from brian.inspection import modified_variables, namespace
    from brian.optimiser import freeze
    from brian.utils.separate_equations import separate_equations
    from brian.experimental.codegen.c_support_code import *
else:
    from ..utils.documentation import flattened_docstring
    from ..stdunits import ms
    from ..connections import Connection, DelayConnection
    from ..log import log_debug, log_warn
    from ..monitor import RecentStateMonitor, SpikeMonitor
    from ..network import NetworkOperation
    from ..neurongroup import NeuronGroup
    from ..equations import Equations
    from new_c_propagate import *
    from ..inspection import modified_variables, namespace
    from codegen.expressions import *
    from ..optimiser import freeze
    from ..utils.separate_equations import separate_equations
    from codegen.c_support_code import *
from scipy import weave
import re

__all__ = ['CSTDP']


class CSTDP(NetworkOperation):
    def __init__(self, C, eqs, pre, post, wmin=0, wmax=Inf, level=0,
                 clock=None, delay_pre=None, delay_post=None):
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        C.compress()
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
        # Find which ones are directly modified (e.g. regular expression matching; careful with comments)
        vars_pre = [var for var in vars if var in modified_variables(pre)]
        vars_post = [var for var in vars if var in modified_variables(post)]

        # additional dependencies are used to ensure that if there are multiple
        # pre/post separated equations they are grouped together as one
        additional_deps = ['__pre_deps='+'+'.join(vars_pre),
                           '__post_deps='+'+'.join(vars_post)]
        separated_equations = separate_equations(eqs_obj, additional_deps)
        if not len(separated_equations) == 2:
            print separated_equations
            raise ValueError('Equations should separate into pre and postsynaptic variables.')
        sep_pre, sep_post = separated_equations
        for v in vars_pre:
            if v in sep_post._diffeq_names:
                sep_pre, sep_post = sep_post, sep_pre
                break

        index_pre = [i for i in range(len(vars)) if vars[i] in vars_pre or vars[i] in sep_pre._diffeq_names]
        index_post = [i for i in range(len(vars)) if vars[i] in vars_post or vars[i] in sep_post._diffeq_names]

        vars_pre = array(vars)[index_pre].tolist()
        vars_post = array(vars)[index_post].tolist()

        # Check pre/post consistency
        shared_vars = set(vars_pre).intersection(vars_post)
        if shared_vars != set([]):
            raise Exception, str(list(shared_vars)) + " are both presynaptic and postsynaptic!"

        # Substitute equations/aliases into pre/post code
        def substitute_eqs(code):
            for name in sep_pre._eq_names[-1::-1]+sep_post._eq_names[-1::-1]: # reverse order, as in equations.py
                if name in sep_pre._eq_names:
                    expr = sep_pre._string[name]
                else:
                    expr = sep_post._string[name]
                code = re.sub("\\b" + name + "\\b", '(' + expr + ')', code)
            return code
        pre = substitute_eqs(pre)
        post = substitute_eqs(post)

        # Create namespaces for pre and post codes
        pre_namespace = namespace(pre, level=level + 1)
        post_namespace = namespace(post, level=level + 1)

        def splitcode(incode):
            num_perneuron = num_persynapse = 0
            reordering_warning = False
            incode_lines = [line.strip() for line in incode.split('\n') if line.strip()]
            per_neuron_lines = []
            per_synapse_lines = []
            for line in incode_lines:
                if not line.strip(): continue
                m = re.search(r'\bw\b\s*[^><=]?=', line) # lines of the form w = ..., w *= ..., etc.
                if m:
                    num_persynapse += 1
                    per_synapse_lines.append(line)
                else:
                    num_perneuron += 1
                    if num_persynapse!=0 and not reordering_warning:
                        log_warn('brian.experimental.cstdp', 'STDP operations are being re-ordered, results may be wrong.')
                        reordering_warning = True
                    per_neuron_lines.append(line)
            return per_neuron_lines, per_synapse_lines

        per_neuron_pre, per_synapse_pre = splitcode(pre)
        per_neuron_post, per_synapse_post = splitcode(post)

        all_vars = vars_pre + vars_post + ['w']        

        per_neuron_pre = [c_single_statement(freeze(line, all_vars, pre_namespace)) for line in per_neuron_pre]
        per_neuron_post = [c_single_statement(freeze(line, all_vars, post_namespace)) for line in per_neuron_post]
        per_synapse_pre = [c_single_statement(freeze(line, all_vars, pre_namespace)) for line in per_synapse_pre]
        per_synapse_post = [c_single_statement(freeze(line, all_vars, post_namespace)) for line in per_synapse_post]

        per_neuron_pre = '\n'.join(per_neuron_pre)
        per_neuron_post = '\n'.join(per_neuron_post)
        per_synapse_pre = '\n'.join(per_synapse_pre)
        per_synapse_post = '\n'.join(per_synapse_post)

        # Neuron groups
        G_pre = NeuronGroup(len(C.source), model=sep_pre, clock=self.clock)
        G_post = NeuronGroup(len(C.target), model=sep_post, clock=self.clock)
        G_pre._S[:] = 0
        G_post._S[:] = 0
        self.pre_group = G_pre
        self.post_group = G_post
        var_group = {}
        for i, v in enumerate(vars_pre):
            var_group[v] = G_pre
        for i, v in enumerate(vars_post):
            var_group[v] = G_post
        self.var_group = var_group

        self.contained_objects += [G_pre, G_post]

        vars_pre_ind = {}
        for i, var in enumerate(vars_pre):
            vars_pre_ind[var] = i
        vars_post_ind = {}
        for i, var in enumerate(vars_post):
            vars_post_ind[var] = i

        prevars_dict = dict((k, G_pre.state(k)) for k in vars_pre)
        postvars_dict = dict((k, G_post.state(k)) for k in vars_post)

        clipcode = ''
        if isfinite(wmin):
            clipcode += 'if(w<%wmin%) w = %wmin%;\n'.replace('%wmin%', repr(float(wmin)))
        if isfinite(wmax):
            clipcode += 'if(w>%wmax%) w = %wmax%;\n'.replace('%wmax%', repr(float(wmax)))

        if not isinstance(C, DelayConnection):
            precode = iterate_over_spikes('_j', '_spikes',
                        (load_required_variables('_j', prevars_dict),
                         transform_code(per_neuron_pre),
                         iterate_over_row('_k', 'w', C.W, '_j',
                            (load_required_variables('_k', postvars_dict),
                             transform_code(per_synapse_pre),
                             ConnectionCode(clipcode)))))
            postcode = iterate_over_spikes('_j', '_spikes',
                        (load_required_variables('_j', postvars_dict),
                         transform_code(per_neuron_post),
                         iterate_over_col('_i', 'w', C.W, '_j',
                            (load_required_variables('_i', prevars_dict),
                             transform_code(per_synapse_post),
                             ConnectionCode(clipcode)))))
            log_debug('brian.experimental.c_stdp', 'CSTDP Pre code:\n' + str(precode))
            log_debug('brian.experimental.c_stdp', 'CSTDP Post code:\n' + str(postcode))
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
            pre_updater = SpikeMonitor(C.source, function=precode, delay=delay_pre)
            post_updater = SpikeMonitor(C.target, function=postcode, delay=delay_post)
            updaters = [pre_updater, post_updater]
            self.contained_objects += [pre_updater, post_updater]
        else:
            if delay_pre is not None or delay_post is not None:
                raise ValueError("Must use delay_pre=delay_post=None for the moment.")
            max_delay = C._max_delay * C.target.clock.dt
            # Ensure that the source and target neuron spikes are kept for at least the
            # DelayConnection's maximum delay
            C.source.set_max_delay(max_delay)
            C.target.set_max_delay(max_delay)

            self.G_pre_monitors = {}
            self.G_post_monitors = {}
            self.G_pre_monitors.update(((var, RecentStateMonitor(G_pre, vars_pre_ind[var], duration=(C._max_delay + 1) * C.target.clock.dt, clock=G_pre.clock)) for var in vars_pre))
            self.G_post_monitors.update(((var, RecentStateMonitor(G_post, vars_post_ind[var], duration=(C._max_delay + 1) * C.target.clock.dt, clock=G_post.clock)) for var in vars_post))
            self.contained_objects += self.G_pre_monitors.values()
            self.contained_objects += self.G_post_monitors.values()

            prevars_dict_delayed = dict((k, self.G_pre_monitors[k]) for k in prevars_dict.keys())
            postvars_dict_delayed = dict((k, self.G_post_monitors[k]) for k in postvars_dict.keys())

            precode_immediate = iterate_over_spikes('_j', '_spikes',
                                    (load_required_variables('_j', prevars_dict),
                                     transform_code(per_neuron_pre)))
            precode_delayed = iterate_over_spikes('_j', '_spikes',
                                     iterate_over_row('_k', 'w', C.W, '_j', extravars={'_delay':C.delayvec},
                                        code=(
                                         ConnectionCode('double _t_past = _max_delay-_delay;', vars={'_max_delay':float(max_delay)}),
                                         load_required_variables_pastvalue('_k', '_t_past', postvars_dict_delayed),
                                         transform_code(per_synapse_pre),
                                         ConnectionCode(clipcode))))
            postcode = iterate_over_spikes('_j', '_spikes',
                            (load_required_variables('_j', postvars_dict),
                             transform_code(per_neuron_post),
                             iterate_over_col('_i', 'w', C.W, '_j', extravars={'_delay':C.delayvec},
                                code=(
                                 load_required_variables_pastvalue('_i', '_delay', prevars_dict_delayed),
                                 transform_code(per_synapse_post),
                                 ConnectionCode(clipcode)))))
            log_debug('brian.experimental.c_stdp', 'CSTDP Pre code (immediate):\n' + str(precode_immediate))
            log_debug('brian.experimental.c_stdp', 'CSTDP Pre code (delayed):\n' + str(precode_delayed))
            log_debug('brian.experimental.c_stdp', 'CSTDP Post code:\n' + str(postcode))
            pre_updater_immediate = SpikeMonitor(C.source, function=precode_immediate)
            pre_updater_delayed = SpikeMonitor(C.source, function=precode_delayed, delay=max_delay)
            post_updater = SpikeMonitor(C.target, function=postcode)
            updaters = [pre_updater_immediate, pre_updater_delayed, post_updater]
            self.contained_objects += updaters

    def __call__(self):
        pass

    def __getattr__(self, name):
        if name == 'var_group':
            # this seems mad - the reason is that getattr is only called if the thing hasn't
            # been found using the standard methods of finding attributes, which for var_index
            # should have worked, this is important because the next line looks for var_index
            # and if we haven't got a var_index we don't want to get stuck in an infinite
            # loop
            raise AttributeError
        if not hasattr(self, 'var_group'):
            # only provide lookup of variable names if we have some variable names, i.e.
            # if the var_index attribute exists
            raise AttributeError
        G = self.var_group[name]
        return G.state_(name)

    def __setattr__(self, name, val):
        if not hasattr(self, 'var_group') or name not in self.var_group:
            object.__setattr__(self, name, val)
        else:
            G = self.var_group[name]
            G.state_(name)[:] = val

if __name__ == '__main__':
    from time import time
    log_level_debug()

    structure = 'dense'
    delay = False

    if not delay:
        delay = 0 * ms

    max_delay = 5 * ms
    N = 1000
    taum = 10 * ms
    tau_pre = 20 * ms
    tau_post = tau_pre
    Ee = 0 * mV
    vt = -54 * mV
    vr = -60 * mV
    El = -74 * mV
    taue = 5 * ms
    F = 15 * Hz
    gmax = .01
    dA_pre = .01
    dA_post = -dA_pre * tau_pre / tau_post * 1.05

    eqs_neurons = '''
    dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
    dge/dt=-ge/taue : 1
    '''

    input = PoissonGroup(N, rates=F)
    neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
    synapses = Connection(input, neurons, 'ge', weight=rand(len(input), len(neurons)) * gmax,
                        structure=structure, delay=delay, max_delay=max_delay)
    neurons.v = vr

    #stdp=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)
    ## Explicit STDP rule
    eqs_stdp = '''
    dA_pre/dt=-A_pre/tau_pre : 1
    dA_post/dt=-A_post/tau_post : 1
    '''
    dA_post *= gmax
    dA_pre *= gmax
    stdp = CSTDP(synapses, eqs=eqs_stdp, pre='A_pre+=dA_pre;w+=A_post',
                 post='A_post+=dA_post;w+=A_pre', wmax=gmax)

    rate = PopulationRateMonitor(neurons)

    start_time = time()
    run(100 * second, report='text')
    print "Simulation time:", time() - start_time

    subplot(311)
    plot(rate.times / second, rate.smooth_rate(100 * ms))
    subplot(312)
    plot(synapses.W.todense() / gmax, '.')
    subplot(313)
    hist(synapses.W.todense() / gmax, 20)
    show()
