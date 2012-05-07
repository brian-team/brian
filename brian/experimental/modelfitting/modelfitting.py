from brian import Equations, NeuronGroup, Clock, CoincidenceCounter, Network, zeros, array, \
                    ones, kron, ms, second, concatenate, hstack, sort, nonzero, diff, TimedArray, \
                    reshape, sum, log, Monitor, NetworkOperation, defaultclock, linspace, vstack, \
                    arange, sort_spikes, rint, SpikeMonitor, Connection
from brian.tools.statistics import firing_rate, get_gamma_factor
try:
    from playdoh import *
except Exception, e:
    print e
    raise ImportError("Playdoh must be installed (https://code.google.com/p/playdoh/)")

try:
    import pycuda
    from gpu_modelfitting import GPUModelFitting
    can_use_gpu = True
except ImportError:
    can_use_gpu = False
from brian.experimental.codegen.integration_schemes import *
from criteria import *
from simulator import * 
import sys, cPickle

__all__ = ['modelfitting', 'print_table', 'PSO', 'GA', 'CMAES',
           'slice_trace', 'transform_spikes',
           'MAXCPU', 'MAXGPU',
           'GammaFactor','GammaFactor2', 'LpError','VanRossum','Brette',
           'simulate', 
           'debug_level', 'info_level', 'warning_level', 'open_server']




class ModelFitting(Fitness):
    def initialize(self, **kwds):
        
        # Gets the key,value pairs in shared_data
        for key, val in self.shared_data.iteritems():
            setattr(self, key, val)
            
        # Gets the key,value pairs in **kwds
        for key, val in kwds.iteritems():
            setattr(self, key, val)
        
        self.model = cPickle.loads(self.model)
        if type(self.model) is str:
            self.model = Equations(self.model)
        
        self.simulator = Simulator(self.model, self.reset, self.threshold,
                                   inputs = self.inputs, input_var = self.input_var,
                                   dt = self.dt,
                                   refractory = self.refractory,
                                   max_refractory = self.max_refractory,
                                   spikes = self.spikes,
                                   traces = self.traces,
                                   groups = self.groups,
                                   slices = self.slices,
                                   overlap = self.overlap,
                                   onset = self.onset,
                                   neurons = self.nodesize,
                                   initial_values = self.initial_values,
                                   unit_type = self.unit_type,
                                   stepsize = self.stepsize,
                                   precision = self.precision,
                                   criterion = self.criterion,
                                   ntrials=self.ntrials,
                                   method = self.method
                                   )
            
    def evaluate(self, **param_values):
        """
        Use fitparams['delays'] to take delays into account
        Use fitparams['refractory'] to take refractory into account
        """
        values = self.simulator.run(**param_values)
        return values




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
                 slices=1,
                 overlap=None,
                 onset=None,
                 initial_values=None,
                 stepsize=100 * ms,
                 unit_type=None,
                 total_units=None,
                 ntrials=1,
                 cpu=None,
                 gpu=None,
                 precision='double', # set to 'float' or 'double' to specify single or double precision on the GPU
                 machines=[],
                 allocation=None,
                 returninfo=False,
                 scaling=None,
                 algorithm=CMAES,
                 async = None,
                 criterion=None,
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
        if (param not in model._diffeq_names) and (param != 'delays') and (param != 'tau_metric'):
            raise Exception("Parameter %s must be defined as a parameter in the model" % param)
    
    if criterion is None:
        criterion = GammaFactor()
        
    data = array(data)
    if criterion.type == 'spikes':
        # Make sure that 'data' is a N*2-array
        if data.ndim == 1:
            data = concatenate((zeros((len(data), 1)), data.reshape((-1, 1))), axis=1)
        spikes = data
        traces = None
        if ntrials>1:
            groups = 1
        else:
            groups = int(array(data)[:, 0].max() + 1) # number of target trains

    elif criterion.type == 'trace':
        if data.ndim == 1:
            data = data.reshape((1,-1))
        spikes = None
        traces = data
        groups = data.shape[0]
    elif criterion.type == 'both':
        # TODO
        log_warn("Not implemented yet")
        pass
    inputs = input
    if inputs.ndim==1:
        inputs = inputs.reshape((1,-1))

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')

    # default overlap when no time slicing
    if overlap is None:
        overlap = 0*ms
    if onset is None:
        onset = overlap
#    if slices == 1:
#        overlap = 0*ms
#        onset = overlap
        
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
    else:
        max_refractory = None
    
#    duration = len(input) * dt # duration of the input

    # keyword arguments for Modelfitting initialize
    kwds = dict(   model=cPickle.dumps(model),
                   threshold=threshold,
                   reset=reset,
                   refractory=refractory,
                   max_refractory=max_refractory,
                   input_var=input_var, dt=dt,
                   criterion=criterion,
                   slices=slices,
                   overlap=overlap,
                   returninfo=returninfo,
                   precision=precision,
                   stepsize=stepsize,
                  ntrials=ntrials,
                   method=method,
                   onset=onset)

    shared_data = dict(inputs=inputs,
                       traces=traces,
                       spikes=spikes,
                       initial_values=initial_values)

    if async:
        r = maximize_async( ModelFitting,
                            shared_data=shared_data,
                            kwds = kwds,
                            groups=groups,
                            popsize=popsize,
                            maxiter=maxiter,
                            optparams=optparams,
                            unit_type = unit_type,
                            machines=machines,
                            allocation=allocation,
                            total_units = total_units,
                            cpu=cpu,
                            gpu=gpu,
                            returninfo=returninfo,
                            codedependencies=[],
                            algorithm=algorithm,
                            scaling=scaling,
                            **params)
    else:
        r = maximize(       ModelFitting,
                            shared_data=shared_data,
                            kwds = kwds,
                            groups=groups,
                            popsize=popsize,
                            maxiter=maxiter,
                            optparams=optparams,
                            unit_type = unit_type,
                            machines=machines,
                            allocation=allocation,
                            total_units = total_units,
                            cpu=cpu,
                            gpu=gpu,
                            returninfo=returninfo,
                            codedependencies=[],
                            algorithm=algorithm,
                            scaling=scaling,
                            **params)

    # r is (results, fitinfo) or (results)
    return r




#def get_spikes(model=None, reset=None, threshold=None,
#                input=None, input_var='I', dt=None,
#                **params):
#    """
#    Retrieves the spike times corresponding to the best parameters found by
#    the modelfitting function.
#    
#    **Arguments**
#    
#    ``model``, ``reset``, ``threshold``, ``input``, ``input_var``, ``dt``
#        Same parameters as for the ``modelfitting`` function.
#        
#    ``**params``
#        The best parameters returned by the ``modelfitting`` function.
#    
#    **Returns**
#    
#    ``spiketimes``
#        The spike times of the model with the given input and parameters.
#    """
#    duration = len(input) * dt
#    ngroups = len(params[params.keys()[0]])
#
#    group = NeuronGroup(N=ngroups, model=model, reset=reset, threshold=threshold,
#                        clock=Clock(dt=dt))
#    group.set_var_by_array(input_var, TimedArray(input, clock=group.clock))
#    for param, values in params.iteritems():
#        if (param == 'delays') | (param == 'fitness'):
#            continue
#        group.state(param)[:] = values
#
#    M = SpikeMonitor(group)
#    net = Network(group, M)
#    net.run(duration)
#    reinit_default_clock()
#    return M.spikes
#
#def predict(model=None, reset=None, threshold=None,
#            data=None, delta=4 * ms,
#            input=None, input_var='I', dt=None,
#            **params):
#    """
#    Predicts the gamma factor of a fitted model with respect to the data with
#    a different input current.
#    
#    **Arguments**
#    
#    ``model``, ``reset``, ``threshold``, ``input_var``, ``dt``
#        Same parameters as for the ``modelfitting`` function.
#        
#    ``input``
#        The input current, that can be different from the current used for the fitting
#        procedure.
#    
#    ``data``
#        The experimental spike times to compute the gamma factor against. They have
#        been obtained with the current ``input``.
#    
#    ``**params``
#        The best parameters returned by the ``modelfitting`` function.
#    
#    **Returns**
#    
#    ``gamma``
#        The gamma factor of the model spike trains against the data.
#        If there were several groups in the fitting procedure, it is a vector
#        containing the gamma factor for each group.
#    """
#    spikes = get_spikes(model=model, reset=reset, threshold=threshold,
#                        input=input, input_var=input_var, dt=dt,
#                        **params)
#
#    ngroups = len(params[params.keys()[0]])
#    gamma = zeros(ngroups)
#    for i in xrange(ngroups):
#        spk = [t for j, t in spikes if j == i]
#        gamma[i] = gamma_factor(spk, data, delta, normalize=True, dt=dt)
#    if len(gamma) == 1:
#        return gamma[0]
#    else:
#        return gamma




if __name__ == '__main__':
    
    from brian import loadtxt, ms, savetxt, loadtxt, Equations, NeuronGroup, run, SpikeMonitor,\
         StateMonitor, Network
    from pylab import *
    
    def generate_data():
        g = NeuronGroup(1, model=equations, reset=0, threshold=1)
        g.I = TimedArray(input, dt=.1*ms)
        g.tau = 25*ms
        g.R = 3e9
        SpM = SpikeMonitor(g)
        StM = StateMonitor(g, 'V', record=True)
        net = Network(g, SpM, StM)
        net.run(1*second)
        return StM.values[0], SpM.spikes
    
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
        #tau_metric : second
    ''')
    input = loadtxt('current.txt')
    
    # ARTIFICIAL DATA: R=3e9, tau=25*ms
#    spikes = loadtxt('spikes.txt') # real data
    trace, spikes = generate_data()
#    savetxt('trace_artificial.txt', trace)
#    savetxt('spikes_artificial.txt', spikes)
    
#    trace = loadtxt('trace_artificial.txt')
    overlap = 10*ms
    slices = 1
    dt = .1*ms
    input, trace = slice_trace(input, trace, slices = slices, dt=dt, overlap = overlap)
    
    # GAMMA FACTOR
#    criterion = GammaFactor(delta=2*ms)
#    spikes= loadtxt('spikes_artificial.txt')
#    data = spikes
#    data[:,1] += 50*ms
    
#    # LP ERROR
    criterion = LpError(p=2, varname='V')
    data = trace
    print data.shape
    print input.shape 
#    # Van Rossum
#    criterion = VanRossum(tau=2*ms)
#    data = spikes
    
    #Brette
#    criterion = Brette()
#    data = spikes

    results = modelfitting( model = equations,
                            reset = 0,
                            threshold = 1,
                            data = data,
                            input = input,
                            #onset = overlap,
                            gpu = 1,
                            #cpu=4,
                            dt = dt,
                            popsize = 20000,
                            maxiter = 10,
                            criterion = criterion,
                            R = [1.0e9,1.0e9, 9.0e9, 9.0e9],
                            tau = [10*ms,10*ms, 40*ms, 40*ms],
                            algorithm=CMAES,
                            tau_metric= [0.5*ms,0.5*ms, 4*ms, 4*ms],
                            #delays=[-1*ms, 1*ms]
                            )
    print_table(results)