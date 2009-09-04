from brian import *
from nose.tools import *
from brian.utils.approximatecomparisons import is_approx_equal, is_within_absolute_tolerance

def test():
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

    The :class:`StateMonitor` object has the following properties (where names
    without an underscore return :class:`QuantityArray` objects with appropriate
    units and names with an underscore return array objects without
    units):

    times, times_
        The times at which recordings were made
    mean, mean_
        The mean value of the state variable for every neuron in the
        group (not just the ones specified in the record keyword)
    var, var_
        The unbiased estimate of the variances, as in mean
    std, std_
        The square root of var, as in mean
        
    In addition, if M is a :class:`StateMonitor` object, you write::
    
        M[i]
    
    for the recorded values of neuron i (if it was specified with the
    record keyword). It returns a :class:`QuantityArray` object with units. Downcast
    to an array without units by writing asarray(M[i]).
    
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

    spikes = [(0,3*ms),(1,4*ms),(0,7*ms)]
    
    G = SpikeGeneratorGroup(2,spikes)
    M = SpikeMonitor(G)
    net = Network(G,M)
    net.run(10*ms)
    
    assert (M.nspikes==3)
    for (mi, mt), (i, t) in zip(M.spikes,spikes):
        assert (mi==i)
        assert (is_approx_equal(mt,t))
    
    # test that SpikeMonitor function calling usage does what you'd expect    
    
    f_spikes = []
    
    def f(spikes):
        if len(spikes):
            f_spikes.extend(spikes)
    
    G = SpikeGeneratorGroup(2,spikes)
    M = SpikeMonitor(G,function=f)
    net = Network(G,M)
    reinit_default_clock()
    net.run(10*ms)
    assert (f_spikes==[0,1,0])
    
    # test interface for StateMonitor object
    
    dV = 'dV/dt = 0*Hz : 1.'
    G = NeuronGroup(3,model=dV,reset=0.,threshold=10.)
    @network_operation(when='start')
    def f(clock):
        if clock.t>=1*ms:
            G.V = [1.,2.,3.]
    M1 = StateMonitor(G,'V')
    M2 = StateMonitor(G,'V',record=0)
    M3 = StateMonitor(G,'V',record=[0,1])
    M4 = StateMonitor(G,'V',record=True)
    reinit_default_clock()
    net = Network(G,f,M1,M2,M3,M4)
    net.run(2*ms)
    assert (is_within_absolute_tolerance(M2[0][0],0.))
    assert (is_within_absolute_tolerance(M2[0][-1],1.))
    assert (is_within_absolute_tolerance(M3[1][0],0.))
    assert (is_within_absolute_tolerance(M3[1][-1],2.))
    assert (is_within_absolute_tolerance(M4[2][0],0.))
    assert (is_within_absolute_tolerance(M4[2][-1],3.))
    assert_raises(IndexError,M1.__getitem__,0)
    assert_raises(IndexError,M2.__getitem__,1)
    assert_raises(IndexError,M3.__getitem__,2)
    assert_raises(IndexError,M4.__getitem__,3)
    for M in [M3, M4]:
        assert (is_within_absolute_tolerance(float(max(abs(M.times-M2.times))),float(0*ms)))
        assert (is_within_absolute_tolerance(float(max(abs(M.times_-M2.times_))),0.))
    for M in [M2, M3, M4]:
        assert (is_within_absolute_tolerance(float(max(abs(M.mean-M1.mean))),0.))
        assert (is_within_absolute_tolerance(float(max(abs(M.var-M1.var))),0.))
        assert (is_within_absolute_tolerance(float(max(abs(M.std-M1.std))),0.))
        assert (is_within_absolute_tolerance(float(max(abs(M.mean_-M1.mean_))),0.))
        assert (is_within_absolute_tolerance(float(max(abs(M.var_-M1.var_))),0.))
        assert (is_within_absolute_tolerance(float(max(abs(M.std_-M1.std_))),0.))
    assert (is_within_absolute_tolerance(float(M2.times[0]),float(0*ms)))
    d = diff(M2.times)
    assert (is_within_absolute_tolerance(max(d),min(d)))
    assert (is_within_absolute_tolerance(float(max(d)),float(get_default_clock().dt)))
    # construct unbiased estimator from variances of recorded arrays
    v = qarray([ var(M4[0]), var(M4[1]), var(M4[2]) ]) * float(len(M4[0])) / float(len(M4[0])-1)
    m = qarray([0.5, 1.0, 1.5])
    assert (is_within_absolute_tolerance(abs(max(M1.mean-m)),0.))
    assert (is_within_absolute_tolerance(abs(max(M1.var-v)),0.))
    assert (is_within_absolute_tolerance(abs(max(M1.std-v**0.5)),0.))
    
    # test when, timestep, clock for StateMonitor
    c = Clock(dt=0.1*ms)
    cslow = Clock(dt=0.2*ms)
    dV = 'dV/dt = 0*Hz : 1.'
    G = NeuronGroup(1,model=dV,reset=0.,threshold=1.,clock=c)
    @network_operation(when='start',clock=c)
    def f():
        G.V = 2.
    M1 = StateMonitor(G,'V',record=True,clock=cslow)
    M2 = StateMonitor(G,'V',record=True,timestep=2,clock=c)
    M3 = StateMonitor(G,'V',record=True,when='before_groups',clock=c)
    net = Network(G,f,M1,M2,M3,M4)
    net.run(2*ms)
    assert (2*len(M1[0])==len(M3[0]))
    assert (len(M1[0])==len(M2[0]))
    for i in range(len(M1[0])):
        assert (is_within_absolute_tolerance(M1[0][i],M2[0][i]))
        assert (is_within_absolute_tolerance(M1[0][i],0.))
    for x in M3[0]:
        assert (is_within_absolute_tolerance(x,2.))
        
    reinit_default_clock() # for next test

if __name__=='__main__':
    test()