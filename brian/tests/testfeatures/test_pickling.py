import pickle

from brian import *

def test_pickling():
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

    #very simple test of pickling for a NeuronGroup and a Network
    G = NeuronGroup(42, model=LazyStateUpdater())
    net = Network(G)
    pickled = pickle.dumps(net)
    unpickled = pickle.loads(pickled)
    assert(len(unpickled) == 42)

if __name__ == '__main__':
    test_pickling()

    