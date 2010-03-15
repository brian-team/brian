from brian import *
from brian.utils.statistics import firing_rate
from clusterfitting import *
from brian.library.modelfitting.clustertools import *
try:
    import pycuda
    can_use_gpu = True
except ImportError:
    can_use_gpu = False

__all__ = ['modelfitting', 'modelfitting_worker']

def modelfitting(model = None, reset = None, threshold = None,
                 data = None, 
                 input_var = 'I', input = None, dt = None,
                 particles = 1000, iterations = 10, pso_params = None,
                 delta = 2*ms, includedelays = True,
                 slices = 1, overlap = None,
                 initial_values = None,
                 verbose = True, stepsize = 100*ms,
                 use_gpu = None, max_cpu = None, max_gpu = None,
                 precision=None, # set to 'float' or 'double' to specify single or double precision on the GPU
                 machines = [], named_pipe = None, port = None, authkey='brian cluster tools',
                 returninfo = False,
                 **params):
    '''
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
    ``particles``
        Number of particles per target train used by the particle swarm optimization algorithm.
    ``iterations``
        Number of iterations in the particle swarm optimization algorithm.
    ``pso_params``
        Parameters of the PSO algorithm. It is a list with three scalar values (omega, c_l, c_g).
        The parameter ``omega`` is the "inertial constant", ``c_l`` is the "local best"
        constant affecting how much the particle's personl best influences its movement, and
        ``c_g`` is the "global best" constant affecting how much the global best
        position influences each particle's movement. See the
        `wikipedia entry on PSO <http://en.wikipedia.org/wiki/Particle_swarm_optimization>`__
        for more details (note that they use ``c_1`` and ``c_2`` instead of ``c_l``
        and ``c_g``). Reasonable values are (.9, .5, 1.5), but experimentation
        with other values is a good idea.
    ``delta=2*ms``
        The precision factor delta (a scalar value in second).
    ``includedelays=True``
        A boolean indicating whether optimizing the spike delay or not.
    ``initial_values``
        A dictionary containing the initial values for the state variables.
    ``verbose=True``
        A boolean value indicating whether printing the progress of the optimization algorithm or not.
    ``use_gpu``
        A boolean value indicating whether using the GPU or not. This value is not taken into account
        if no GPU is present on the computer.
    ``max_cpu``
        The maximum number of CPUs to use in parallel. It is set to the number of CPUs in the machine by default.
    ``max_gpu``
        The maximum number of GPUs to use in parallel. It is set to the number of GPUs in the machine by default.
    ``precision``
        A string set to either ``float`` or ``double`` to specify whether to use
        single or double precision on the GPU. If it is not specified, it will
        use the best precision available.
    ``machines=[]``
        A list of machine names to use in parallel. See :ref:`modelfitting-clusters`.
    ``named_pipe``
        Set to ``True`` to use Windows named pipes for networking, or a string
        to use a particular name for the pipe. See :ref:`modelfitting-clusters`.
    ``port``
        The port number for IP networking, you only need to specify this if the
        default value of 2718 is blocked. See :ref:`modelfitting-clusters`.
    ``return_matrix``
        Set it to ``True`` to return the matrix of the fitness values for all particles and all iterations.
    
    **Return values**
    
    ``best_params``
        A :class:`~brian.Parameters` object containing the best parameter values for each target spike train
        found by the optimization algorithm. ``best_params[param_name]`` is a vector containing
        the parameter values for each target.
    ``best_values``
        A vector containing the best gamma factor values for each target.
        For more details on the gamma factor, see
        `Jolivet et al. 2008, "A benchmark test for a quantitative assessment of simple neuron models", J. Neurosci. Methods <http://www.ncbi.nlm.nih.gov/pubmed/18160135>`__ (available in PDF
        `here <http://icwww.epfl.ch/~gerstner/PUBLICATIONS/Jolivet08.pdf>`__).
    ``fitness_matrices``
        If ``return_matrix`` is set to ``True``, ``fitness_matrices[i]`` is a (N*iterations) matrix
        containing the histogram of the fitness values among particle within each group at each 
        iteration of the optimization algorithm.
    '''

    # Use GPU ?
    if can_use_gpu & (use_gpu is not False):
        use_gpu = True
        gpu_policy = 'prefer_gpu'
    else:
        use_gpu = False
        gpu_policy = 'no_gpu'

    # TODO: no time slicing yet
    slices = 1

    # Make sure that 'data' is a N*2-array
    data = array(data)
    if data.ndim == 1:
        data = concatenate((zeros((len(data), 1)), data.reshape((-1,1))), axis=1)

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    
    # default overlap when no time slicing
    if slices == 1:
        overlap = 0*ms

    # common values
    group_size = particles # Number of particles per target train
    group_count = int(array(data)[:,0].max()+1) # number of target trains
    N = group_size*group_count # number of neurons
    duration = len(input)*dt # duration of the input

    # Prepares the data
    i, t = zip(*data)
    i = array(i)
    t = array(t)
    alls = []
    n = 0
    pointers = []
    target_length = zeros(group_count)
    target_rates = zeros(group_count)
    for j in range(group_count):
        s = sort(t[i==j])
        target_length[j] = len(s)
        target_rates[j] = firing_rate(s)
        s = hstack((-1*second, s, duration+1*second))
        alls.append(s)
        pointers.append(n)
        n += len(s)
    spiketimes = hstack(alls)
    pointers = array(pointers, dtype=int)
    model_target = array(kron(arange(group_count), ones(group_size)), dtype=int)
    spiketimes_offset = pointers[model_target]
    spikedelays = zeros(N)
    target_length = kron(target_length, ones(group_size))
    target_rates = kron(target_rates, ones(group_size))

    # WARNING: PSO-specific
    optparams = pso_params 
    if optparams is None:
        optparams = [.9, .5, 1.5]

    # TODO: convert an Expression to the original string
#    if isinstance(model, Equations):
#        model = model.expr
    shared_data = dict(model = model, # MUST be a string
                       threshold = threshold,
                       reset = reset,
                       input_var = input_var,
                       input = input,
                       dt = dt,
                       duration = duration,
                       spiketimes = spiketimes,
                       group_size = group_size,
                       group_count = group_count,
                       delta = delta,
                       returninfo = returninfo,
                       initial_values = initial_values,
                       onset = 0*ms,
                       fitparams = params,
                       optparams = optparams)
    
    local_data = dict(spiketimes_offset = spiketimes_offset,
                      target_length = target_length,
                      target_rates = target_rates)

    cluster_info = dict(gpu_policy = gpu_policy,
                        max_cpu = max_cpu,
                        max_gpu = max_gpu,
                        machines = machines,
                        named_pipe = named_pipe,
                        port = port,
                        authkey = authkey)
    
    fm = FittingManager(shared_data, local_data, iterations, cluster_info)
    fm.run()
    results, fitinfo = fm.get_results()
    fm.print_results()

    if returninfo:
        return results, fitinfo
    else:
        return results

def modelfitting_worker(max_gpu=None, max_cpu=None, port=None, named_pipe=None,
                        authkey='brian cluster tools'):
    '''
    Model fitting worker script. See documentation in :ref:`model-fitting-library`.
    '''
    cluster_worker_script(light_worker,
                          max_gpu=max_gpu, max_cpu=max_cpu, port=port,
                          named_pipe=named_pipe, authkey=authkey)

if __name__ == '__main__':
    model = '''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    '''
    threshold = 1
    reset = 0
    
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')
#    spikes = [(0, spike*second) for spike in spikes0]
#    spikes.extend([(1, spike*second+5*ms) for spike in spikes0])
    
    results = modelfitting(model = model, reset = reset, threshold = threshold, 
                                 data = spikes, 
                                 input = input, dt = .1*ms,
                                 max_cpu = 4,
                                 particles = 1000, iterations = 3, delta = 2*ms,
                                 R = [1.0e9, 1.0e10],
                                 tau = [1*ms, 50*ms],
                                 _delays = [-10*ms, 10*ms])

