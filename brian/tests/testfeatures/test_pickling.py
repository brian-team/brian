import pickle
import itertools

from numpy.testing import assert_equal

from brian import *

def test_timed_array_pickling():
    '''    
    Tests pickling of brian objects    
    '''
    
    def timed_array_correctly_pickled(obj):
        '''
        Pickles and unpickles a TimedArray object (using a string, i.e. not
        writing to a file. Returns whether the two objects are identical or not.
        '''
        pickled = pickle.dumps(obj)
        unpickled = pickle.loads(pickled)
        assert((unpickled == obj).all()) # tests the array content
        assert(unpickled.clock.t == obj.clock.t and
               unpickled.clock.dt == obj.clock.dt)
        assert(unpickled.guessed_clock == obj.guessed_clock)
        assert((unpickled.times is None and obj.times is None) or 
                 (unpickled.times == obj.times).all())
        assert(unpickled._dt == obj._dt)
        assert(unpickled._t_init == obj._t_init)        
    
    # Test TimedArray with a clock
    ta = TimedArray([0, 1], clock=Clock(dt=0.2*ms))
    timed_array_correctly_pickled(ta)
    
    # Test TimedArray with the default clock
    ta = TimedArray([0, 1])
    timed_array_correctly_pickled(ta)

def test_neurongroup_pickling():
    #very simple test of pickling for a NeuronGroup and a Network
    G = NeuronGroup(42, model=LazyStateUpdater())
    net = Network(G)
    pickled = pickle.dumps(net)
    unpickled = pickle.loads(pickled)
    assert(len(unpickled) == 42)

def test_linearstateupdater_pickling():
    G = NeuronGroup(10, model='''dv/dt = -(v + I)/ (10 * ms) : volt
                                 I : volt''')
    pickled = pickle.dumps(G)
    unpickled = pickle.loads(pickled)
    assert len(unpickled) == 10
    assert_equal(G._state_updater.A, unpickled._state_updater.A)
    assert G._state_updater.B == unpickled._state_updater.B

def test_synapses_pickling():
    # Test pickling a Network with a Synapses object
    G1 = NeuronGroup(42, model='v:1', threshold='v>1', reset='v=0')
    G1.name = 'G1'
    G2 = NeuronGroup(42, model='v:1')
    G2.name = 'G2'
    G1.v = 1.1
    S = Synapses(G1, G2, model='w:1', pre='v+=w')
    S[:, :] = 'i==j'
    S.w = 'i'
    net = Network(G1, G2, S)
    pickled = pickle.dumps(net)
    net.run(defaultclock.dt)
    assert_equal(G2.v[:], np.arange(len(G1)))
    unpickled_net = pickle.loads(pickled)
    unpickled_G1 = next(group for group in unpickled_net.groups
                        if getattr(group, 'name', None) == 'G1')
    unpickled_G2 = next(group for group in unpickled_net.groups
                        if getattr(group, 'name', None) == 'G2')
    unpickled_S = next(group for group in unpickled_net.groups
                        if isinstance(group, Synapses))
    defaultclock.t = 0*ms
    assert len(unpickled_S) == len(S)
    assert len(unpickled_G1) == len(G1)
    assert len(unpickled_G2) == len(G2)
    unpickled_net.run(defaultclock.dt)
    assert_equal(unpickled_G2.v[:], np.arange(len(unpickled_G1)))


if __name__ == '__main__':
    test_timed_array_pickling()
    test_neurongroup_pickling()
    test_linearstateupdater_pickling()
    test_synapses_pickling()
