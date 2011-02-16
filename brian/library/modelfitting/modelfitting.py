from brian import Equations, NeuronGroup, Clock, CoincidenceCounter, Network, zeros, array, \
                    ones, kron, ms, second, concatenate, hstack, sort, nonzero, diff, TimedArray, \
                    reshape, sum, log
from brian.tools.statistics import firing_rate, get_gamma_factor
try:
    from playdoh import *
except Exception, e:
    print e
    raise ImportError("Playdoh must be installed (https://code.google.com/p/playdoh/)")

try:
    import pycuda
    from brian.library.modelfitting.gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False
from brian.experimental.codegen.integration_schemes import *
import sys, cPickle

__all__ = ['modelfitting', 'print_table', 'get_spikes', 'predict', 'PSO', 'GA','CMAES',
           'MAXCPU', 'MAXGPU',
           'debug_level', 'info_level', 'warning_level', 'open_server']


class ModelFitting(Fitness):
    def initialize(self, **kwds):
        self.use_gpu = self.unit_type=='GPU'
        # Gets the key,value pairs in shared_data
        for key, val in self.shared_data.iteritems():
            setattr(self, key, val)
        # Gets the key,value pairs in **kwds
        for key, val in kwds.iteritems():
            setattr(self, key, val)

#        log_info(self.model)
        self.model = cPickle.loads(self.model)
#        log_info(self.model)

        # if model is a string
        if type(self.model) is str:
            self.model = Equations(self.model)

        self.total_steps = int(self.duration / self.dt)

        self.neurons = self.nodesize
        self.groups = self.groups

        # Time slicing
        self.input = self.input[0:self.slices * (len(self.input) / self.slices)] # makes sure that len(input) is a multiple of slices
        self.duration = len(self.input) * self.dt # duration of the input
        self.sliced_steps = len(self.input) / self.slices # timesteps per slice
        self.overlap_steps = int(self.overlap / self.dt) # timesteps during the overlap
        self.total_steps = self.sliced_steps + self.overlap_steps # total number of timesteps
        self.sliced_duration = self.overlap + self.duration / self.slices # duration of the vectorized simulation
        self.N = self.neurons * self.slices # TOTAL number of neurons in this worker

        self.input = hstack((zeros(self.overlap_steps), self.input)) # add zeros at the beginning because there is no overlap from the previous slice

        # Prepares data (generates I_offset, spiketimes, spiketimes_offset)
        self.prepare_data()

        # Add 'refractory' parameter on the CPU on the CPU only
        if not self.use_gpu:
            if self.max_refractory is not None:
                refractory = 'refractory'
                self.model.add_param('refractory', second)
            else:
                refractory = self.refractory
        else:
            if self.max_refractory is not None:
                refractory = 0*ms
            else:
                refractory = self.refractory

        # Must recompile the Equations : the functions are not transfered after pickling/unpickling
        self.model.compile_functions()

        self.group = NeuronGroup(self.N,
                                 model=self.model,
                                 reset=self.reset,
                                 threshold=self.threshold,
                                 refractory=refractory,
                                 max_refractory = self.max_refractory,
                                 method = self.method,
                                 clock=Clock(dt=self.dt))
        
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value

        # Injects current in consecutive subgroups, where I_offset have the same value
        # on successive intervals
        k = -1
        for i in hstack((nonzero(diff(self.I_offset))[0], len(self.I_offset) - 1)):
            I_offset_subgroup_value = self.I_offset[i]
            I_offset_subgroup_length = i - k
            sliced_subgroup = self.group.subgroup(I_offset_subgroup_length)
            input_sliced_values = self.input[I_offset_subgroup_value:I_offset_subgroup_value + self.total_steps]
            sliced_subgroup.set_var_by_array(self.input_var, TimedArray(input_sliced_values, clock=self.group.clock))
            k = i

        if self.use_gpu:
            # Select integration scheme according to method
            if self.method == 'Euler': scheme = euler_scheme
            elif self.method == 'RK': scheme = rk2_scheme
            elif self.method == 'exponential_Euler': scheme = exp_euler_scheme
            else: raise Exception("The numerical integration method is not valid")
            
            self.mf = GPUModelFitting(self.group, self.model, self.input, self.I_offset,
                                      self.spiketimes, self.spiketimes_offset, zeros(self.neurons), 0*ms, self.delta,
                                      precision=self.precision, scheme=scheme)
        else:
            self.cc = CoincidenceCounter(self.group, self.spiketimes, self.spiketimes_offset,
                                        onset=self.onset, delta=self.delta)

    def prepare_data(self):
        """
        Generates I_offset, spiketimes, spiketimes_offset from data,
        and also target_length and target_rates.
        
        The neurons are first grouped by time slice : there are group_size*group_count
        per group/time slice
        Within each time slice, the neurons are grouped by target train : there are
        group_size neurons per group/target train
        """
        # Generates I_offset
        self.I_offset = zeros(self.N, dtype=int)
        for slice in range(self.slices):
            self.I_offset[self.neurons * slice:self.neurons * (slice + 1)] = self.sliced_steps * slice

        # Generates spiketimes, spiketimes_offset, target_length, target_rates
        i, t = zip(*self.data)
        i = array(i)
        t = array(t)
        alls = []
        n = 0
        pointers = []
        dt = self.dt

        target_length = []
        target_rates = []
        model_target = []
        group_index = 0

        neurons_in_group = self.subpopsize
        for j in xrange(self.groups):
#            neurons_in_group = self.groups[j] # number of neurons in the current group and current worker
            s = sort(t[i == j])
            target_length.extend([len(s)] * neurons_in_group)
            target_rates.extend([firing_rate(s)] * neurons_in_group)

            for k in xrange(self.slices):
            # first sliced group : 0...0, second_train...second_train, ...
            # second sliced group : first_train_second_slice...first_train_second_slice, second_train_second_slice...
                spikeindices = (s >= k * self.sliced_steps * dt) & (s < (k + 1) * self.sliced_steps * dt) # spikes targeted by sliced neuron number k, for target j
                targeted_spikes = s[spikeindices] - k * self.sliced_steps * dt + self.overlap_steps * dt # targeted spikes in the "local clock" for sliced neuron k
                targeted_spikes = hstack((-1 * second, targeted_spikes, self.sliced_duration + 1 * second))
                model_target.extend([k + group_index * self.slices] * neurons_in_group)
                alls.append(targeted_spikes)
                pointers.append(n)
                n += len(targeted_spikes)
            group_index += 1
        pointers = array(pointers, dtype=int)
        model_target = array(hstack(model_target), dtype=int)

        self.spiketimes = hstack(alls)
        self.spiketimes_offset = pointers[model_target] # [pointers[i] for i in model_target]

        # Duplicates each target_length value 'group_size' times so that target_length[i]
        # is the length of the train targeted by neuron i
        self.target_length = array(target_length, dtype=int)
        self.target_rates = array(target_rates)

    def evaluate(self, **param_values):
        """
        Use fitparams['delays'] to take delays into account
        Use fitparams['refractory'] to take refractory into account
        """
        delays = param_values.pop('delays', zeros(self.neurons))
        refractory = param_values.pop('refractory', zeros(self.neurons))

        # kron spike delays
        delays = kron(delays, ones(self.slices))
        refractory = kron(refractory, ones(self.slices))

        # Sets the parameter values in the NeuronGroup object
        self.group.reinit()
        for param, value in param_values.iteritems():
            self.group.state(param)[:] = kron(value, ones(self.slices)) # kron param_values if slicing
            
        # Reinitializes the model variables
        if self.initial_values is not None:
            for param, value in self.initial_values.iteritems():
                self.group.state(param)[:] = value

        if self.use_gpu:
            # Reinitializes the simulation object
            self.mf.reinit_vars(self.input, self.I_offset, self.spiketimes, self.spiketimes_offset, delays, refractory)
            # LAUNCHES the simulation on the GPU
            self.mf.launch(self.duration, self.stepsize)
            coincidence_count = self.mf.coincidence_count
            spike_count = self.mf.spike_count
        else:
            # set the refractory period
            if self.max_refractory is not None:
                self.group.refractory = refractory
            
            self.cc = CoincidenceCounter(self.group, self.spiketimes, self.spiketimes_offset,
                                        onset=self.onset, delta=self.delta)
            # Sets the spike delay values
            self.cc.spikedelays = delays
            # Reinitializes the simulation objects
            self.group.clock.reinit()
#            self.cc.reinit()
            net = Network(self.group, self.cc)
            # LAUNCHES the simulation on the CPU
            net.run(self.duration)
            coincidence_count = self.cc.coincidences
            spike_count = self.cc.model_length

        coincidence_count = sum(reshape(coincidence_count, (self.slices, -1)), axis=0)
        spike_count = sum(reshape(spike_count, (self.slices, -1)), axis=0)

        gamma = get_gamma_factor(coincidence_count, spike_count, self.target_length, self.target_rates, self.delta)

        return gamma




def modelfitting(model=None,
                 reset=None,
                 threshold=None,
                 refractory=0*ms,
                 data=None,
                 input_var='I',
                 input=None,
                 dt=None,
                 popsize=1000,
                 maxiter=10,
                 delta=4*ms,
                 slices=1,
                 overlap=0*second,
                 initial_values=None,
                 stepsize=100 * ms,
                 unit_type=None,
                 total_units=None,
                 cpu=None,
                 gpu=None,
                 precision='double', # set to 'float' or 'double' to specify single or double precision on the GPU
                 machines=[],
                 allocation=None,
                 returninfo=False,
                 scaling=None,
                 algorithm=CMAES,
                 async = None,
                 optparams={},
                 method='Euler',
                 **params):
    """
    Model fitting function.
    
    Fits a spiking neuron model to electrophysiological data (injected current and spikes).
    
    See also the section :ref:`model-fitting-library` in the user manual.
    
    **Arguments**
    
    ``model``
        An :class:`~brian.Equations` object containing the equations defining the model.
    
    ``reset``
        A reset value for the membrane potential, or a string containing the reset
        equations.
        
    ``threshold``
        A threshold value for the membrane potential, or a string containing the threshold
        equations.
        
    ``refractory``
        The refractory period in second. If it's a single value, the same refractory will be
        used in all the simulations. If it's a list or a tuple, the fitting will also
        optimize the refractory period (see ``**params`` below).
        
        Warning: when using a refractory period, you can't use a custom reset, only a fixed one.
        
    ``data``
        A list of spike times, or a list of several spike trains as a list of pairs (index, spike time)
        if the fit must be performed in parallel over several target spike trains. In this case,
        the modelfitting function returns as many parameters sets as target spike trains.
        
    ``input_var='I'``
        The variable name used in the equations for the input current.
        
    ``input``
        A vector of values containing the time-varying signal the neuron responds to (generally
        an injected current).
        
    ``dt``
        The time step of the input (the inverse of the sampling frequency).
        
    ``**params``
        The list of parameters to fit the model with. Each parameter must be set as follows:
        ``param_name=[bound_min, min, max, bound_max]``
        where ``bound_min`` and ``bound_max`` are the boundaries, and ``min`` and ``max``
        specify the interval from which the parameter values are uniformly sampled at
        the beginning of the optimization algorithm.
        If not using boundaries, set ``param_name=[min, max]``.
        
        Also, you can add a fit parameter which is a spike delay for all spikes :
        add the special parameter ``delays`` in ``**params``, for example 
        ``modelfitting(..., delays=[-10*ms, 10*ms])``.
        
        You can also add fit the refractory period by specifying 
        ``modelfitting(..., refractory=[-10*ms, 10*ms])``.
        
    ``popsize``
        Size of the population (number of particles) per target train used by the optimization algorithm.
        
    ``maxiter``
        Number of iterations in the optimization algorithm.
        
    ``optparams``
        Optimization algorithm parameters. It is a dictionary: keys are parameter names,
        values are parameter values or lists of parameters (one value per group). 
        This argument is specific to the optimization
        algorithm used. See :class:`PSO`, :class:`GA`, :class:`CMAES`. 
        
    ``delta=4*ms``
        The precision factor delta (a scalar value in second).
        
    ``slices=1``
        The number of time slices to use.
        
    ``overlap=0*ms``
        When using several time slices, the overlap between consecutive slices, in seconds.
        
    ``initial_values``
        A dictionary containing the initial values for the state variables.
        
    ``cpu``
        The number of CPUs to use in parallel. It is set to the number of CPUs in the machine by default.
        
    ``gpu``
        The number of GPUs to use in parallel. It is set to the number of GPUs in the machine by default.
        
    ``precision``
        GPU only: a string set to either ``float`` or ``double`` to specify whether to use
        single or double precision on the GPU. If it is not specified, it will
        use the best precision available.
        
    ``returninfo=False``
        Boolean indicating whether the modelfitting function should return technical information
        about the optimization.
        
    ``scaling=None``
        Specify the scaling used for the parameters during the optimization. 
        It can be ``None`` or ``'mapminmax'``. It is ``None``
        by default (no scaling), and ``mapminmax`` by default for the CMAES algorithm.
        
    ``algorithm=CMAES``
        The optimization algorithm. It can be :class:`PSO`, :class:`GA` or :class:`CMAES`.
         
    ``optparams={}``
         Optimization parameters. See
         
    ``method='Euler'``
        Integration scheme used on the CPU and GPU: ``'Euler'`` (default), ``RK``, 
        or ``exponential_Euler``.
        See also :ref:`numerical-integration`.
        
    ``machines=[]``
        A list of machine names to use in parallel. See :ref:`modelfitting-clusters`.
    
    **Return values**
    
    Return an :class:`OptimizationResult` object with the following attributes:
    
    ``best_pos``
        Minimizing position found by the algorithm. For array-like fitness functions,
        it is a single vector if there is one group, or a list of vectors.
        For keyword-like fitness functions, it is a dictionary
        where keys are parameter names and values are numeric values. If there are several groups,
        it is a list of dictionaries.
    
    ``best_fit``
        The value of the fitness function for the best positions. It is a single value if 
        there is one group, or it is a list if there are several groups.
    
    ``info``
        A dictionary containing various information about the optimization.

    Also, the following syntax is possible with an ``OptimizationResult`` instance ``or``.
    The ``key`` is either an optimizing parameter name for keyword-like fitness functions,
    or a dimension index for array-like fitness functions.
    
    ``or[key]``
        it is the best ``key`` parameter found (single value), or the list
        of the best parameters ``key`` found for all groups.
    
    ``or[i]``
        where ``i`` is a group index. This object has attributes ``best_pos``, ``best_fit``,
        ``info`` but only for group ``i``.
    
    ``or[i][key]``
        where ``i`` is a group index, is the same as ``or[i].best_pos[key]``.

    For more details on the gamma factor, see
    `Jolivet et al. 2008, "A benchmark test for a quantitative assessment of simple neuron models", J. Neurosci. Methods <http://www.ncbi.nlm.nih.gov/pubmed/18160135>`__ (available in PDF
    `here <http://icwww.epfl.ch/~gerstner/PUBLICATIONS/Jolivet08.pdf>`__).
    """
    
    for param in params.keys():
        if (param not in model._diffeq_names) and (param != 'delays'):
            raise Exception("Parameter %s must be defined as a parameter in the model" % param)

    # Make sure that 'data' is a N*2-array
    data = array(data)
    if data.ndim == 1:
        data = concatenate((zeros((len(data), 1)), data.reshape((-1, 1))), axis=1)

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')

    # default overlap when no time slicing
    if slices == 1:
        overlap = 0 * ms

    # default allocation
    if cpu is None and gpu is None and unit_type is None:
        if CANUSEGPU: unit_type = 'GPU'
        else: unit_type = 'CPU'

    # check numerical integration method
    if (gpu>0 or unit_type == 'GPU') and method not in ['Euler', 'RK', 'exponential_Euler']:
        raise Exception("The method can only be 'Euler', 'RK', or 'exponential_Euler' when using the GPU") 
    if method not in ['Euler', 'RK', 'exponential_Euler', 'linear', 'nonlinear']:
        raise Exception("The method can only be 'Euler', 'RK', 'exponential_Euler', 'linear', or 'nonlinear'")

    if (algorithm == CMAES) & (scaling is None):
        scaling = 'mapminmax'
        
    # determines whether optimization over refractoriness or not
    if type(refractory) is tuple or type(refractory) is list:
        params['refractory'] = refractory
        max_refractory = refractory[-1]
#        refractory = 'refractory'
    else:
        max_refractory = None

    # common values
#    group_size = particles # Number of particles per target train
    groups = int(array(data)[:, 0].max() + 1) # number of target trains
#    N = group_size * group_count # number of neurons
    duration = len(input) * dt # duration of the input    

    # keyword arguments for Modelfitting initialize
    kwds = dict(   model=cPickle.dumps(model),
                   threshold=threshold,
                   reset=reset,
                   refractory=refractory,
                   max_refractory=max_refractory,
                   input_var=input_var,dt=dt,
                   duration=duration,delta=delta,
                   slices=slices,
                   overlap=overlap,
                   returninfo=returninfo,
                   precision=precision,
                   stepsize=stepsize,
                   method=method,
                   onset=0 * ms)

    shared_data = dict(input=input,
                       data=data,
                       initial_values=initial_values)

    if async:
        r = maximize_async(   ModelFitting,
                        shared_data=shared_data,
                        kwds = kwds,
                        groups=groups,
                        popsize=popsize,
                        maxiter=maxiter,
                        optparams=optparams,
                        unit_type = unit_type,
                        total_units = total_units,
                        machines=machines,
                        allocation=allocation,
                        cpu=cpu,
                        gpu=gpu,
                        returninfo=returninfo,
                        codedependencies=[],
                        algorithm=algorithm,
                        scaling=scaling,
                        **params)
    else:
        r = maximize(   ModelFitting,
                        shared_data=shared_data,
                        kwds = kwds,
                        groups=groups,
                        popsize=popsize,
                        maxiter=maxiter,
                        optparams=optparams,
                        unit_type = unit_type,
                        total_units = total_units,
                        machines=machines,
                        allocation=allocation,
                        cpu=cpu,
                        gpu=gpu,
                        returninfo=returninfo,
                        codedependencies=[],
                        algorithm=algorithm,
                        scaling=scaling,
                        **params)

    return r




def get_spikes(model=None, reset=None, threshold=None,
                input=None, input_var='I', dt=None,
                **params):
    """
    Retrieves the spike times corresponding to the best parameters found by
    the modelfitting function.
    
    **Arguments**
    
    ``model``, ``reset``, ``threshold``, ``input``, ``input_var``, ``dt``
        Same parameters as for the ``modelfitting`` function.
        
    ``**params``
        The best parameters returned by the ``modelfitting`` function.
    
    **Returns**
    
    ``spiketimes``
        The spike times of the model with the given input and parameters.
    """
    duration = len(input) * dt
    ngroups = len(params[params.keys()[0]])

    group = NeuronGroup(N=ngroups, model=model, reset=reset, threshold=threshold,
                        clock=Clock(dt=dt))
    group.set_var_by_array(input_var, TimedArray(input, clock=group.clock))
    for param, values in params.iteritems():
        if (param == 'delays') | (param == 'fitness'):
            continue
        group.state(param)[:] = values

    M = SpikeMonitor(group)
    net = Network(group, M)
    net.run(duration)
    reinit_default_clock()
    return M.spikes

def predict(model=None, reset=None, threshold=None,
            data=None, delta=4 * ms,
            input=None, input_var='I', dt=None,
            **params):
    """
    Predicts the gamma factor of a fitted model with respect to the data with
    a different input current.
    
    **Arguments**
    
    ``model``, ``reset``, ``threshold``, ``input_var``, ``dt``
        Same parameters as for the ``modelfitting`` function.
        
    ``input``
        The input current, that can be different from the current used for the fitting
        procedure.
    
    ``data``
        The experimental spike times to compute the gamma factor against. They have
        been obtained with the current ``input``.
    
    ``**params``
        The best parameters returned by the ``modelfitting`` function.
    
    **Returns**
    
    ``gamma``
        The gamma factor of the model spike trains against the data.
        If there were several groups in the fitting procedure, it is a vector
        containing the gamma factor for each group.
    """
    spikes = get_spikes(model=model, reset=reset, threshold=threshold,
                        input=input, input_var=input_var, dt=dt,
                        **params)

    ngroups = len(params[params.keys()[0]])
    gamma = zeros(ngroups)
    for i in xrange(ngroups):
        spk = [t for j, t in spikes if j == i]
        gamma[i] = gamma_factor(spk, data, delta, normalize=True, dt=dt)
    if len(gamma) == 1:
        return gamma[0]
    else:
        return gamma
