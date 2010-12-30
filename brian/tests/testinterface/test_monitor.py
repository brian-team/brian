from brian import *
from nose.tools import *
from brian.utils.approximatecomparisons import is_approx_equal, is_within_absolute_tolerance
try:
    from brian.experimental.cuda.gpu_modelfitting import *
    import pycuda.autoinit as autoinit
    use_gpu = True
except ImportError:
    use_gpu = False

def test_spikemonitor():
    '''
    :class:`SpikeMonitor`
    ~~~~~~~~~~~~~~~~~~~~~
    
    Records spikes from a :class:`NeuronGroup`. Initialised as one of::
    
        SpikeMonitor(source(,record=True))
        SpikeMonitor(source,function=function)
    
    Where:
    
    source
        A :class:`NeuronGroup` to record from
    record
        True or False to record all the spikes or just summary
        statistics.
    function
        A function f(spikes) which is passed the array of spikes
        numbers that have fired called each step, to define
        custom spike monitoring.
    
    Has two attributes:
    
    nspikes
        The number of recorded spikes
    spikes
        A time ordered list of pairs (i,t) where neuron i fired
        at time t.
    
    :class:`StateMonitor`
    ~~~~~~~~~~~~~~~~~~~~~
    
    Records the values of a state variable from a :class:`NeuronGroup`.
    Initialise as::
    
        StateMonitor(P,varname(,record=False)
            (,when='end)(,timestep=1)(,clock=clock))
    
    Where:
    
    P
        The group to be recorded from
    varname
        The state variable name or number to be recorded
    record
        What to record. The default value is False and the monitor will
        only record summary statistics for the variable. You can choose
        record=integer to record every value of the neuron with that
        number, record=list of integers to record every value of each of
        those neurons, or record=True to record every value of every
        neuron (although beware that this may use a lot of memory).
    when
        When the recording should be made in the :class:`Network` update, possible
        values are any of the strings: 'start', 'before_groups', 'after_groups',
        'before_connections', 'after_connections', 'before_resets',
        'after_resets', 'end' (in order of when they are run).
    timestep
        A recording will be made each timestep clock updates (so timestep
        should be an integer).
    clock
        A clock for the update schedule, use this if you have specified a
        clock other than the default one in your network, or to update at a
        lower frequency than the update cycle. Note though that if the clock
        here is different from the main clock, the when parameter will not
        be taken into account, as network updates are done clock by clock.
        Use the timestep parameter if you need recordings to be made at a
        precise point in the network update step.

    The :class:`StateMonitor` object has the following properties:

    times
        The times at which recordings were made
    mean
        The mean value of the state variable for every neuron in the
        group (not just the ones specified in the record keyword)
    var
        The unbiased estimate of the variances, as in mean
    std
        The square root of var, as in mean
        
    In addition, if M is a :class:`StateMonitor` object, you write::
    
        M[i]
    
    for the recorded values of neuron i (if it was specified with the
    record keyword). It returns an array object.
    
    Others
    ~~~~~~
    
    The following monitors also exist, but are not part of the
    assured interface because their syntax is subject to change. See the documentation
    for each class for more details.
    
    * :class:`Monitor` (base class)
    * :class:`ISIHistogramMonitor`
    * :class:`FileSpikeMonitor`
    * :class:`PopulationRateMonitor`
    '''
    reinit_default_clock()

    # test that SpikeMonitor retrieves the spikes generator by SpikeGeneratorGroup

    spikes = [(0, 3 * ms), (1, 4 * ms), (0, 7 * ms)]

    G = SpikeGeneratorGroup(2, spikes, clock=defaultclock)
    M = SpikeMonitor(G)
    net = Network(G, M)
    net.run(10 * ms)

    assert (M.nspikes == 3)
    for (mi, mt), (i, t) in zip(M.spikes, spikes):
        assert (mi == i)
        assert (is_approx_equal(mt, t))

    # test that SpikeMonitor function calling usage does what you'd expect    

    f_spikes = []

    def f(spikes):
        if len(spikes):
            f_spikes.extend(spikes)

    G = SpikeGeneratorGroup(2, spikes, clock=defaultclock)
    M = SpikeMonitor(G, function=f)
    net = Network(G, M)
    reinit_default_clock()
    net.run(10 * ms)
    assert (f_spikes == [0, 1, 0])

    # test interface for StateMonitor object

    dV = 'dV/dt = 0*Hz : 1.'
    G = NeuronGroup(3, model=dV, reset=0., threshold=10.)
    @network_operation(when='start')
    def f(clock):
        if clock.t >= 1 * ms:
            G.V = [1., 2., 3.]
    M1 = StateMonitor(G, 'V')
    M2 = StateMonitor(G, 'V', record=0)
    M3 = StateMonitor(G, 'V', record=[0, 1])
    M4 = StateMonitor(G, 'V', record=True)
    reinit_default_clock()
    net = Network(G, f, M1, M2, M3, M4)
    net.run(2 * ms)
    assert (is_within_absolute_tolerance(M2[0][0], 0.))
    assert (is_within_absolute_tolerance(M2[0][-1], 1.))
    assert (is_within_absolute_tolerance(M3[1][0], 0.))
    assert (is_within_absolute_tolerance(M3[1][-1], 2.))
    assert (is_within_absolute_tolerance(M4[2][0], 0.))
    assert (is_within_absolute_tolerance(M4[2][-1], 3.))
    assert_raises(IndexError, M1.__getitem__, 0)
    assert_raises(IndexError, M2.__getitem__, 1)
    assert_raises(IndexError, M3.__getitem__, 2)
    assert_raises(IndexError, M4.__getitem__, 3)
    for M in [M3, M4]:
        assert (is_within_absolute_tolerance(float(max(abs(M.times - M2.times))), float(0 * ms)))
        assert (is_within_absolute_tolerance(float(max(abs(M.times_ - M2.times_))), 0.))
    assert (is_within_absolute_tolerance(float(M2.times[0]), float(0 * ms)))
    d = diff(M2.times)
    assert (is_within_absolute_tolerance(max(d), min(d)))
    assert (is_within_absolute_tolerance(float(max(d)), float(get_default_clock().dt)))
    # construct unbiased estimator from variances of recorded arrays
    v = array([ var(M4[0]), var(M4[1]), var(M4[2]) ]) * float(len(M4[0])) / float(len(M4[0]) - 1)
    m = array([0.5, 1.0, 1.5])
    assert (is_within_absolute_tolerance(abs(max(M1.mean - m)), 0.))
    assert (is_within_absolute_tolerance(abs(max(M1.var - v)), 0.))
    assert (is_within_absolute_tolerance(abs(max(M1.std - v ** 0.5)), 0.))

    # test when, timestep, clock for StateMonitor
    c = Clock(dt=0.1 * ms)
    cslow = Clock(dt=0.2 * ms)
    dV = 'dV/dt = 0*Hz : 1.'
    G = NeuronGroup(1, model=dV, reset=0., threshold=1., clock=c)
    @network_operation(when='start', clock=c)
    def f():
        G.V = 2.
    M1 = StateMonitor(G, 'V', record=True, clock=cslow)
    M2 = StateMonitor(G, 'V', record=True, timestep=2, clock=c)
    M3 = StateMonitor(G, 'V', record=True, when='before_groups', clock=c)
    net = Network(G, f, M1, M2, M3, M4)
    net.run(2 * ms)
    print M1[0], M3[0]
    assert (2 * len(M1[0]) == len(M3[0]))
    assert (len(M1[0]) == len(M2[0]))
    for i in range(len(M1[0])):
        assert (is_within_absolute_tolerance(M1[0][i], M2[0][i]))
        assert (is_within_absolute_tolerance(M1[0][i], 0.))
    for x in M3[0]:
        assert (is_within_absolute_tolerance(x, 2.))

    reinit_default_clock() # for next test

def test_coincidencecounter():
    """
    Simulates an IF model with constant input current and checks
    the total number of coincidences with prediction.
    """

    eqs = """
    dV/dt = (-V+R*I)/tau : 1
    I : 1
    R : 1
    tau : second
    """
    reset = 0
    threshold = 1

    duration = 500 * ms
    input = 1.2 + .2 * randn(int(duration / defaultclock._dt))
    delta = 4 * ms
    n = 10

    def get_data(n):
        # Generates data from an IF neuron
        group = NeuronGroup(N=1, model=eqs, reset=reset, threshold=threshold,
                            method='Euler', refractory=3 * delta)
        group.I = TimedArray(input, start=0 * second, dt=defaultclock.dt)
        group.R = 1.0
        group.tau = 20 * ms
        M = SpikeMonitor(group)
        stM = StateMonitor(group, 'V', record=True)
        net = Network(group, M, stM)
        net.run(duration)
        data = M.spikes
#        train0 = M.spiketimes[0]
        reinit_default_clock()

#        trains = []
#        for i in range(n):
#            trains += zip(i*ones(len(train0), dtype='int'), (array(train0) + (delta*c) * rand(len(train0))))
#        trains.sort(lambda x,y: (2*int(x[1]>y[1])-1))
#        trains = [(i,t*second) for i,t in trains]

        return data, stM.values#, trains

    data, data_voltage = get_data(n=n)
    train0 = [t for i, t in data]

    group = NeuronGroup(n, eqs, reset=reset, threshold=threshold,
                        method='Euler')
    group.I = TimedArray(input, start=0 * second, dt=defaultclock.dt)
    group.R = 1.0 * ones(n)
    group.tau = 20 * ms * (1 + .1 * (2 * rand(n) - 1))

    cc = CoincidenceCounter(source=group, data=([-1 * second] + train0 + [duration + 1 * second]), delta=delta)
    sm = SpikeMonitor(group)
    statem = StateMonitor(group, 'V', record=True)
    net = Network(group, cc, sm, statem)
    net.run(duration)
    reinit_default_clock()

    cpu_voltage = statem.values

    online_coincidences = cc.coincidences
    cpu_spike_count = array([len(sm[i]) for i in range(n)])
    offline_coincidences = array([gamma_factor(sm[i], train0, delta=delta, normalize=False, dt=defaultclock.dt) for i in range(n)])

    if use_gpu:
        # Compute gamma factor with GPU
        inp = array(input)
        I_offset = zeros(n, dtype=int)
        #spiketimes = array(hstack(([-1*second],train0,[data[-1][1]+1*second])))
        spiketimes = array(hstack(([-1 * second], train0, [duration + 1 * second])))
        spiketimes_offset = zeros(n, dtype=int)
        spikedelays = zeros(n)
        cd = CoincidenceCounter(source=group, data=data, delta=delta)
        group.V = 0.0

        mf = GPUModelFitting(group, Equations(eqs),
                             inp, I_offset, spiketimes, spiketimes_offset,
                             spikedelays, delta)

        # Normal GPU launch
        mf.launch(duration)

        # GPU record of voltage and spikes 
    #    allV = []
    #    oldnc = 0
    #    oldsc = 0
    #    allcoinc = []
    #    all_pst = []
    #    all_nst = []
    #    allspike = []
    #    all_nsa = []
    #    all_lsa = []
    #    
    #    for i in xrange(int(duration/defaultclock.dt)):
    #        mf.kernel_func(int32(i), int32(i+1),
    #                         *mf.kernel_func_args, **mf.kernel_func_kwds)
    #        autoinit.context.synchronize()
    #        allV.append(mf.state_vars['V'].get())
    #        all_pst.append(mf.spiketimes.get()[mf.spiketime_indices.get()])
    #        all_nst.append(mf.spiketimes.get()[mf.spiketime_indices.get()+1])
    #        all_nsa.append(mf.next_spike_allowed_arr.get()[0])
    #        all_lsa.append(mf.last_spike_allowed_arr.get()[0])
    #    #        self.next_spike_allowed_arr = gpuarray.to_gpu(ones(N, dtype=bool))
    #    #        self.last_spike_allowed_arr = gpuarray.to_gpu(zeros(N, dtype=bool))
    #        nc = mf.coincidence_count[0]
    #        if nc>oldnc:
    #            oldnc = nc
    #            allcoinc.append(i*defaultclock.dt)
    #        sc = mf.spike_count[0]
    #        if sc>oldsc:
    #            oldsc = sc
    #            allspike.append(i*defaultclock.dt)
    #    
    #    gpu_voltage = array(allV)


        cc = mf.coincidence_count
        gpu_spike_count = mf.spike_count
        cd._model_length = gpu_spike_count
        cd._coincidences = cc
        gpu_coincidences = cc

    print "Spike count"
    print "Data", len(data)
    print "CPU", cpu_spike_count
    if use_gpu:
        print "GPU", gpu_spike_count
        print "max error : %.1f" % max(abs(cpu_spike_count - gpu_spike_count))
        print
    print "Offline"
    print offline_coincidences
    print
    print "Online"
    print online_coincidences
    print "max error : %.6f" % max(abs(online_coincidences - offline_coincidences))
    if use_gpu:
        print
        print "GPU"
        print gpu_coincidences
        print "max error : %.6f" % max(abs(gpu_coincidences - offline_coincidences))

        bad_neuron = nonzero(abs(gpu_coincidences - offline_coincidences) > 1e-10)[0]
        if len(bad_neuron) > 0:
            print "Bad neuron", bad_neuron, group.tau[bad_neuron[0]]
    print
    print
    return

#    plot(linspace(0,duration/second,len(data_voltage[0])), data_voltage[0], 'k', linewidth=.5)
#    plot(linspace(0,duration/second,len(cpu_voltage[0])), cpu_voltage[0], 'b')
#    plot(linspace(0,duration/second,len(gpu_voltage[:,0])), gpu_voltage[:,0], 'g')
#    show()
    times = linspace(0, duration / second, len(all_pst))

    figure()

    plot(times, array(all_pst) * defaultclock.dt)
    plot(times, array(all_nst) * defaultclock.dt)
    plot(train0, train0, 'o')
    plot(allspike, allspike, 'x')
    plot(allcoinc, allcoinc, '+')
    plot(times, array(all_nsa) * times, '--')
    plot(times, array(all_lsa) * times, '-.')
    predicted_spikes = allspike
    target_spikes = train0 + [duration + 1 * second]
    i = 0
    truecoinc = []
    for pred_t in predicted_spikes:
         while target_spikes[i] <= pred_t + delta - 1e-10 * second:
             if abs(target_spikes[i] - pred_t) <= delta + 1e-10 * second:
                 truecoinc.append((pred_t, target_spikes[i]))
                 i += 1
                 break
             i += 1
    print 'Truecoinc:', len(truecoinc)
    for t1, t2 in truecoinc:
        plot([t1, t2], [t1, t2], ':', color=(0.5, 0, 0), lw=3)

    show()

#    assert is_within_absolute_tolerance(online_gamma1,offline_gamma1)    
#    assert is_within_absolute_tolerance(online_gamma2,offline_gamma2)   


#def test_vectorized_spikemonitor():
#    eqs = """
#    dV/dt = (-V+I)/tau : 1
#    tau : second
#    I : 1
#    """ 
#    N = 30
#    taus = 10*ms + 90*ms * rand(N)
#    duration = 1000*ms
#    input = 2.0 + 3.0 * rand(int(duration/defaultclock._dt))
#    vgroup = VectorizedNeuronGroup(model=eqs, reset=0, threshold=1,
#                                   input=input, slices=2, overlap=200*ms, tau=taus)
#    M = SpikeMonitor(vgroup)
#    run(vgroup.duration)
#    raster_plot(M)
#    show()

if __name__ == '__main__':
    test_spikemonitor()
#    test_coincidencecounter()
