from brian import *
from nose.tools import assert_raises

def test_construction():
    ''' Test various ways of constructing a TimedArray. '''
    
    reinit_default_clock()
    
    ar = array([0, 2, 4, 6])
    
    def assert_values(ta, ar, start, dt):
        # check values at timepoints
        for index, val in enumerate(ar):
            assert ta(start + dt * index) == val
        
        # check values between timepoints
        for index, val in enumerate(ar[:-1]):
            assert ta(start + dt * (index + 0.4)) == val
            
    # defaultclock
    ta = TimedArray(ar)
    assert ta.clock == defaultclock
    assert_values(ta, ar, 0 * second, defaultclock.dt)
    
    # a given clock
    clock = Clock(dt=0.25 * ms)
    ta = TimedArray(ar, clock=clock)
    assert ta.clock == clock
    assert_values(ta, ar, 0 * second, clock.dt)
    
    # given dt
    dt = 0.5 * ms
    ta = TimedArray(ar, dt=dt)
    assert_values(ta, ar, 0 * second, dt=dt)
    
    # given start and dt
    start, dt = 1 * ms, 0.5 * ms
    ta = TimedArray(ar, start=start, dt=dt)
    assert_values(ta, ar, start=start, dt=dt)

    # giving start, dt and clock should not work
    assert_raises(ValueError, lambda : TimedArray(ar, start=start, dt=dt,
                                                  clock=Clock()))

    # a list of times
    dt = 0.25 * ms
    times = arange(len(ar)) * dt
    ta = TimedArray(ar, times=times)
    
    # list of times + clock should not work
    assert_raises(ValueError, lambda : TimedArray(ar, times=times,
                                                  clock=Clock()))

def test_access():
    ''' Test various ways of accessing a TimedArray. '''
    
    reinit_default_clock()
    
    # 1-D array
    ar = array([0, 2, 4, 6])
    ta = TimedArray(ar)
    assert ta[1] == array(ar[1])
    
    assert (ta[:] == ar[:]).all() and (ta.times == ta[:].times).all()
    assert (ta[:2] == ar[:2]).all()
    assert (ta[:2].times == ta.times[:2]).all() 
    index_array = array([0, 2])
    assert (ta[index_array] == ar[index_array]).all()
    assert (ta[index_array].times == ta.times[index_array]).all()

    times = arange(len(ar)) * defaultclock.dt
    assert (ta(times) == ar).all()
    
    # values beyond the last timepoint should default to the last value
    assert ta(1 * second) == ar[-1]
    
    # 2-D array
    ar = array([[0, 1], [2, 3], [4, 5], [6, 7]])
    ta = TimedArray(ar)
    assert len(ta(0 * second)) == 2
    
    # last values
    assert (ta(1 * second) == ar[-1, :]).all()

    # this should use the first time for the first "neuron" and the second time 
    # for the second one
    assert (ta(array([0 * second, 2 * defaultclock.dt])) == array([ar[0, 0], ar[2, 1]])).all()
    
    
def test_neurongroup_integration():
    ''' Test using a TimedArray for setting values of a NeuronGroup. '''
    
    reinit_default_clock()
    
    duration = 5 * ms
    
    ar = arange(0, int(duration / defaultclock.dt)) * 3    
    ta = TimedArray(ar)
    
    # Link variable to TimedArray via TimedArraySetter
    G = NeuronGroup(1, model='p : 1')
    G.p = ta
    mon = StateMonitor(G, 'p', record=0)
    run(duration)
    
    assert (mon[0] == ar).all()

    reinit_default_clock()
    # Using explicit times
    ta = TimedArray(ar, times=arange(int(duration / defaultclock.dt)) * defaultclock.dt)
    G = NeuronGroup(1, model='p : 1')
    G.p = ta
    mon = StateMonitor(G, 'p', record=0)
    run(duration)

    assert (mon[0] == ar).all()
        
    reinit_default_clock()

    # use an array with custom, irregular times
    ar = array([1, 5])
    ta = TimedArray(ar, times=array([0*ms, 3 * ms]))
    G = NeuronGroup(1, model='p : 1')
    G.p = ta
    mon = StateMonitor(G, 'p', record=0)    
    run(duration)

    assert (mon[0][mon.times < 3 * ms] == 1).all()
    assert (mon[0][mon.times >= 3 * ms] == 5).all()
    
    # TODO: Check 2-D arrays as well

if __name__ == '__main__':
    test_construction()
    test_access()
    test_neurongroup_integration()
