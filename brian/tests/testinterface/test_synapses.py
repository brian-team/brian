'''
Tests various ways of constructing synapses.
'''
# Needed so that python 2.5 does not complain about a SyntaxError
from __future__ import with_statement

import warnings
import sys

from nose.tools import assert_raises
import numpy as np

from brian.network import Network
from brian.clock import defaultclock, reinit_default_clock
from brian.synapses import Synapses, SynapticEquations
from brian.neurongroup import NeuronGroup
from brian.directcontrol import SpikeGeneratorGroup
from brian.monitor import StateMonitor
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
    
    # deleting a synapse should not work
    def delete_synapse():
        syn[0, 0] = False
    assert_raises(ValueError, delete_synapse)
    
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
    
    net = Network(G, syn)
    net.run(defaultclock.dt)
    # adding synapses should not work after the simulation has already been run
    def adding_a_synapse():
        syn[0, 1] = True
    assert_raises(AttributeError, adding_a_synapse)
    
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

    # Calling connect_one_to_one without specifying groups should use the full
    # source or target group
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_one_to_one(subgroup1)
    assert len(syn) == len(subgroup1)

    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_one_to_one(post=subgroup2)
    assert len(syn) == len(subgroup1)

    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_one_to_one()
    assert len(syn) == len(subgroup1)

    # connect_one_to_one should only work with subgroups of the same size
    assert_raises(TypeError, lambda : syn.connect_one_to_one(G[:2], G[:1]))

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

    # Calling connect_random without specifying groups should use the full
    # source or target group
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_random(subgroup1, sparseness=1.0)
    assert len(syn) == 10 * 10

    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_random(post=subgroup2, sparseness=1.0)
    assert len(syn) == 10 * 10

    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn.connect_random(sparseness=1.0)
    assert len(syn) == 10 * 10

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

    # this test requires python 2.6
    if sys.version_info[0] >= 2 and sys.version_info[1] >=6:
        # Running a model with a synapses object with 0 synapses should raise a warning
        syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            syn.compress()
            # Verify some things
            assert len(w) == 1
    
    # use arrays as neuron indices instead of slices or ints
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn[np.arange(5), np.arange(5)] = True
    syn.w[np.arange(5), np.arange(5)] = 2
    syn.delay[np.arange(5), np.arange(5)] = 5 * ms
    assert len(syn) == 5 * 5
    assert all([syn.w[i, j] == 2 for i in xrange(5) for j in xrange(5)])
    assert all([syn.delay[i, j] == 5 * ms for i in xrange(5) for j in xrange(5)])
    assert all([len(syn.w[i, j]) == 0 for i in xrange(5, len(subgroup1))
                for j in xrange(5, len(subgroup2))])
    assert all([len(syn.delay[i, j]) == 0 for i in xrange(5, len(subgroup1))
                for j in xrange(5, len(subgroup2))])

def test_construction_multiple_synapses():
    '''
    Test the construction of synapses with multiple synapses per connection.
    '''
    G = NeuronGroup(20, model='v:1', threshold=NoThreshold())
    subgroup1, subgroup2 = G[:10], G[10:]
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')    

    # create all-to-all connections with three synapses per connection
    syn[:, :] = 3
    assert len(syn) == 3 * 10 * 10
    # set the weights to the same value for all synapses
    syn.w[:, :] = 2
    # set the delays
    syn.delay[:, :] = 1 * ms

    all_weights = np.array([syn.w[i, j, syn_idx] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2)) for syn_idx in xrange(3)])
    assert (all_weights == 2).all()
    all_delays = np.array([syn.delay[i, j, syn_idx] for i in xrange(len(subgroup1))
                         for j in xrange(len(subgroup2)) for syn_idx in xrange(3)])
    assert (all_delays == 1 * ms).all()


def test_construction_and_access():
    '''
    Test various ways of constructing and accessing synapses.
    '''    
    G = NeuronGroup(20, model='v:1', threshold=NoThreshold())    
    subgroup1, subgroup2 = G[:10], G[10:]
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')    
    
    # single synaptic indices
    syn[3, 4] = True
    syn.w[3, 4] = 0.25
    syn.delay[3, 4] = 1 * ms
    for i in xrange(len(subgroup1)):
        for j in xrange(len(subgroup2)):
            if i == 3 and j == 4:
                assert syn.w[i, j] == 0.25
                assert syn.delay[i, j] == 1 * ms
            else:
                assert len(syn.w[i, j]) == 0
                assert len(syn.delay[i, j]) == 0

    # illegal target index
    def illegal_index():
        syn[3, 10] = True
    assert_raises(ValueError, illegal_index)

    # illegal source index
    def illegal_index(): #@DuplicatedSignature
        syn[10, 4] = True    
    assert_raises(ValueError, illegal_index)
    
    # target slice
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')    
    syn[3, :] = True
    syn.w[3, :] = 0.25
    syn.delay[3, :] = 1 * ms
    for i in xrange(len(subgroup1)):
        for j in xrange(len(subgroup2)):
            if i == 3:
                assert syn.w[i, j] == 0.25
                assert syn.delay[i, j] == 1 * ms
            else:
                assert len(syn.w[i, j]) == 0
                assert len(syn.delay[i, j]) == 0
    # access with slice                
    assert (syn.w[3, :] == 0.25).all()
    assert (syn.delay[3, :] == 1 * ms).all()
    
    # source slice
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')    
    syn[:, 4] = True
    syn.w[:, 4] = 0.25
    syn.delay[:, 4] = 1 * ms
    for i in xrange(len(subgroup1)):
        for j in xrange(len(subgroup2)):
            if j == 4:
                assert syn.w[i, j] == 0.25
                assert syn.delay[i, j] == 1 * ms
            else:
                assert len(syn.w[i, j]) == 0
                assert len(syn.delay[i, j]) == 0
    # access with slice                
    assert (syn.w[:, 4] == 0.25).all()
    assert (syn.delay[:, 4] == 1 * ms).all()
    
    # Use string initialization for synaptic variable
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn[:, 4] = True
    # synapses from even source index have weight 1, uneven --> weight 0.5
    # same for delays (1ms and 0.5ms)
    syn.w[:, 4] = '(i % 2 == 0) * 0.5 + 0.5'
    syn.delay[:, 4] = '(i % 2 == 0) * 0.5 * ms + 0.5 * ms'
    assert (syn.w[::2, 4] == 1).all()
    assert (syn.w[1::2, 4] == 0.5).all()
    assert (syn.delay[::2, 4] == 1 * ms).all()
    assert (syn.delay[1::2, 4] == 0.5 * ms).all()    
    
    # Use string initialization with a numpy function and constant
    syn = Synapses(subgroup1, subgroup2, model='w:1', pre='v += w')
    syn[:, 4] = True
    
    syn.w[:, 4] = '0.5 + 0.5 * np.cos(i / (1.0 * n) * np.pi)'
    syn.delay[:, 4] = '0.0005 + 0.0005 * np.cos(i / (1.0 * n) * np.pi)'
    assert (syn.w[:, 4] == 0.5 + 0.5 * np.cos(np.arange(len(subgroup1)) /
                                              (1.0 * len(subgroup1)) * np.pi)).all()
    # possible delays are always multiples of dt
    delays = 0.5 * ms + 0.5 * ms * np.cos(np.arange(len(subgroup1)) /
                                          (1.0 * len(subgroup1)) * np.pi)
    delays_rounded_to_dt = (delays / defaultclock.dt).astype(np.int) * defaultclock.dt
    assert (syn.delay[:, 4] == delays_rounded_to_dt).all()

def test_model_definition():
    ''' Tests various ways of defining models.
    Note that this test only checks for whether the interface accepts what it
    is supposed to accept, not whether the Synapses class actually does what
    it is supposed to do... '''
    G = NeuronGroup(1, model='v:1', threshold=NoThreshold())
    
    # model, pre and post string
    syn = Synapses(G, model='w:1', pre='v+=w', post='v+=w')
    # list of pre codes
    syn = Synapses(G, model='w:1', pre=['v+=w', 'v+=1'])
    
    # an equation object
    model = SynapticEquations('w:1')
    syn = Synapses(G, model=model, pre='v+=w')
    
    ############################################################################
    # Some more complex examples from the docs
    ############################################################################
    
    # event-driven simulations
    
    model='''w:1
             dApre/dt=-Apre/taupre : 1 (event-driven)
             dApost/dt=-Apost/taupost : 1 (event-driven)'''
    syn = Synapses(G, model=model)
    
    model ='''w : 1
              p : 1'''
    syn = Synapses(G, model=model, pre="v+=w*(rand()<p)")
    
    syn = Synapses(G,
                   model='''x : 1
                            u : 1
                            w : 1''',
                    pre='''u=U+(u-U)*exp(-(t-lastupdate)/tauf)
                           x=1+(x-1)*exp(-(t-lastupdate)/taud)
                           i+=w*u*x
                           x*=(1-u)
                           u+=U*(1-u)''')
    
    # lumped variables
    a = 1 / (10 * ms)
    b = 1 / (20 * ms)
    c = 1 / (30 * ms)
    neurons = NeuronGroup(1, model="""dv/dt=(gtot-v)/(10*ms) : 1
                                  gtot : 1""")
    S=Synapses(neurons,
               model='''dg/dt=-a*g+b*x*(1-g) : 1
                    dx/dt=-c*x : 1
                    w : 1 # synaptic weight
                 ''',
                 pre='x+=w')    
    neurons.gtot = S.g
    
    v0 = 0
    tau = 5 * ms
    neurons = NeuronGroup(1, model='''dv/dt=(v0-v+Igap)/tau : 1
                                      Igap : 1''')
    S=Synapses(neurons, model='''w:1 # gap junction conductance
                                 Igap=w*(v_pre-v_post): 1''')
    neurons.Igap = S.Igap

def test_max_delay():
    '''Test that changing delays after compression works. '''
    
    reinit_default_clock()
    inp = SpikeGeneratorGroup(1, [(0, 1*ms)])
    G = NeuronGroup(1, model='v:1')
    mon = StateMonitor(G, 'v', record=True)
    
    # one synapse
    syn = Synapses(inp, G, model='w:1', pre='v+=w', max_delay=5*ms)
    
    syn[:, :] = 1
    syn.w[:, :] = 1
    syn.delay[:, :] = 0 * ms
    
    net = Network(inp, G, syn, mon)    
    net.run(defaultclock.dt)
    syn.delay[:, :] = 5 * ms    
    net.run(6.5*ms)
    
    # spike should arrive at 5 + 1 ms    
    assert (mon[0][mon.times >= 6 * ms] == 1).all() 
    assert (mon[0][mon.times < 6 * ms] == 0).all()
    
    # same as above but with two synapses
    reinit_default_clock()
    mon.reinit()
    G.reinit()
    
    syn = Synapses(inp, G, model='w:1', pre='v+=w', max_delay=5*ms)
    
    syn[:, :] = 2
    syn.w[:, :] = 1
    syn.delay[:, :] = [0 * ms, 0 * ms]
    
    net = Network(inp, G, syn, mon)    
    net.run(defaultclock.dt)
    syn.delay[:, :] = [5 * ms, 5 * ms]
    net.run(6.5*ms)
    
    # spike should arrive at 5 + 1 ms
    assert (mon[0][mon.times >= 6 * ms] == 2).all()
    assert (mon[0][mon.times < 6 * ms] == 0).all()    
    

################################################################################
# Low level unit tests, test single helper functions
from brian.synapses.synapticvariable import slice_to_array

# avoid nose picking up on slice_to_test as a test_function
from brian.synapses.synapses import slice_to_test as slice_to_t
slice_to_t.__name__ = 'slice_to_t'

from brian.synapses.synapses import invert_array, smallest_inttype, indent


def test_slice_to_array():
    '''
    Test the slice_to_array function (converts a slice to the corresponding
    array of integers).
    '''
    # test special cases: array, sequence, int
    assert (slice_to_array(42) == np.array([42])).all()
    assert (slice_to_array([23, 42]) == np.array([23, 42])).all()
    assert (slice_to_array(np.array([23, 42])) == np.array([23, 42])).all()
    assert (slice_to_array(slice(0, 5)) == np.arange(0, 5)).all()
    assert (slice_to_array(slice(2, 5)) == np.arange(2, 5)).all()
    assert (slice_to_array(slice(2, None), N=5) == np.arange(2, 5)).all()
    assert (slice_to_array(slice(2, 5, 2)) == np.arange(2, 5, 2)).all()
    assert (slice_to_array(slice(2, -1, 2), N=10) == np.arange(2, 9, 2)).all()


def test_slice_to_test():
    '''
    Test the slice_to_test function (converts a slice to a corresponding function,
    returning True in case an index is in the given slice).
    '''    
    testfun = slice_to_t(5)
    assert testfun(5) and not testfun(4)
    
    testfun = slice_to_t(slice(None, 5))
    assert all([testfun(x) for x in xrange(5)])
    assert not testfun(5)
    
    testfun = slice_to_t(slice(0, 5))
    assert all([testfun(x) for x in xrange(5)])
    assert not testfun(5)

    testfun = slice_to_t(slice(0, 10, 2))
    assert all([testfun(x) for x in xrange(0, 10, 2)])
    assert not testfun(1)
    assert not testfun(10)

    testfun = slice_to_t(slice(0, None, 2))
    assert all([testfun(x) for x in xrange(0, 10, 2)])
    assert not testfun(1)


def test_invert_array():
    '''
    Test the invert_array function.
    '''
    assert invert_array(np.array([])) == {}
    
    test_ar = np.array([0, 4, 4, 1, 0, 0])
    inverted_array = invert_array(test_ar)
    # The inverted_array function does not guarantee sorted arrays
    for key in inverted_array:
        inverted_array[key].sort()
    
    assert len(inverted_array) == 3
    assert (inverted_array[0] == np.array([0, 4, 5])).all()
    assert (inverted_array[4] == np.array([1, 2])).all()
    assert (inverted_array[1] == np.array([3])).all()


def test_smallest_inttype():
    '''
    Test the smallest_inttype function (returning the smallest signed integer
    type that can represent a certain number)
    '''
    values = [2**n-1 for n in xrange(1, 36)]
    values.extend([2**n for n in xrange(1, 36)])
    for value in values:
        # only checks for whether the type is big enough, not whether it is the
        # smallest possible type
        inttype = smallest_inttype(value)
        assert inttype(value) == value

def test_indent():
    '''
    Tests the indent function.
    '''
    assert indent('some text') == '    some text'
    assert indent('some text', 2) == '        some text'
    assert indent('    some text') == '        some text'
    before_text='''some text
more text'''
    after_text='''    some text
    more text'''
    assert indent(before_text) == after_text
    
if __name__ == '__main__':
    test_construction_single_synapses()
    test_construction_multiple_synapses()
    test_construction_and_access()
    test_model_definition()
    test_slice_to_array()
    test_slice_to_test()
    test_smallest_inttype()
    test_indent()
    test_max_delay()
