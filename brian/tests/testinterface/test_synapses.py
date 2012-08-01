'''
Tests various ways of constructing synapses.
'''
from nose.tools import assert_raises
import numpy as np

from brian.synapses import Synapses
from brian.neurongroup import NeuronGroup
from brian.threshold import NoThreshold
from brian.stdunits import ms

def test_construction_single_synapses():
    '''
    Test the construction of synapses with a single synapse per connection.
    '''
    G = NeuronGroup(20, model='v:1', threshold=NoThreshold())
    
    # specifying only one group should use it as source and target
    syn = Synapses(G, model='w:1')
    assert syn.source is syn.target

    # specifying source and target with subgroups
    subgroup1, subgroup2 = G[:10], G[10:]
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')    
    assert syn.source is subgroup1
    assert syn.target is subgroup2

    # create all-to-all connections
    syn[:, :] = True
    assert len(syn) == 10 * 10
    # set the weights
    syn.w[:, :] = 2
    # set the delays
    syn.delay[:, :] = 1 * ms

    all_weights = np.array([syn.w[i, j] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2))])
    assert (all_weights == 2).all()
    all_delays = np.array([syn.delay[i, j] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2))])
    assert (all_delays == 1 * ms).all()

    # create one-to-one connections
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn[:, :] = 'i == j'
    syn.w[:, :] = 2
    assert len(syn) == len(subgroup1)
    for i in xrange(len(subgroup1)):
        for j in xrange(len(subgroup2)):
            if i == j:
                assert syn.w[i, j] == 2
            else:                
                assert len(syn.w[i, j]) == 0

    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_one_to_one(subgroup1, subgroup2)
    syn.w[:, :] = 2
    assert len(syn) == len(subgroup1)
    for i in xrange(len(subgroup1)):
        for j in xrange(len(subgroup2)):
            if i == j:
                assert syn.w[i, j] == 2
            else:                
                assert len(syn.w[i, j]) == 0

    # create random connections
    # the only two cases that can be tested exactly are 0 and 100% connection
    # probability
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    # floats are interpreted as connection probabilities
    syn[:, :] = 0.0    
    assert len(syn) == 0
    
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_random(subgroup1, subgroup2, 0.0)    
    assert len(syn) == 0
    
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn[:, :] = 1.0
    assert len(syn) == 10 * 10
    # set the weights
    syn.w[:, :] = 2
    # set the delays
    syn.delay[:, :] = 1 * ms

    all_weights = np.array([syn.w[i, j] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2))])
    assert (all_weights == 2).all()
    all_delays = np.array([syn.delay[i, j] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2))])
    assert (all_delays == 1 * ms).all()
    
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_random(subgroup1, subgroup2, 1.0)
    assert len(syn) == 10 * 10
    # set the weights
    syn.w[:, :] = 2
    # set the delays
    syn.delay[:, :] = 1 * ms

    all_weights = np.array([syn.w[i, j] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2))])
    assert (all_weights == 2).all()
    all_delays = np.array([syn.delay[i, j] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2))])
    assert (all_delays == 1 * ms).all()

    # Just test that probabilities between zero and one work at all
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_random(subgroup1, subgroup2, 0.3)
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn[:, :] = 0.3
    
    # Test that probabilities outside of the legal range raise an error
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    assert_raises(ValueError,
                  lambda : syn.connect_random(subgroup1, subgroup2, 1.3))
    assert_raises(ValueError,
                  lambda : syn.connect_random(subgroup1, subgroup2, -.3))
    def wrong_probability():
        syn[:, :] = -0.3        
    assert_raises(ValueError, wrong_probability)
    def wrong_probability(): #@DuplicatedSignature
        syn[:, :] = 1.3        
    assert_raises(ValueError, wrong_probability)        

if __name__ == '__main__':
    test_construction_single_synapses()
