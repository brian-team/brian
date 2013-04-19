'''
This test covers the to_matrix feature of SynapticVariable
'''
from brian.synapses import *
from brian.neurongroup import *
import numpy as np
import cStringIO
from matplotlib.pyplot import *
from nose.tools import *

def test_first_last():
    # prepare neurongroups for the creation of a synapse object
    N0 = 1
    g0 = NeuronGroup(N0, model = 'dv/dt = 0/ms : 1')
    N1 = 1
    g1 = NeuronGroup(N1, model = 'dv/dt = 0/ms : 1')

    # create two synapses between the same neurons
    synapses = Synapses(g0, g1, model = 'w:1', pre = 'v+=w')
    synapses[0,0] = 2
    synapses.w[0,0,0] = 1
    synapses.w[0,0,1] = 2

    # check the first/last behavior
    w_dense = synapses.w.to_matrix(multiple_synapses = 'first')
    assert w_dense[0,0] == 1
    w_dense = synapses.w.to_matrix(multiple_synapses = 'last')
    assert w_dense[0,0] == 2

def test_min_max_sum():
    # prepare neurongroups for the creation of a synapse object
    N0 = 1
    g0 = NeuronGroup(N0, model = 'dv/dt = 0/ms : 1')
    N1 = 1
    g1 = NeuronGroup(N1, model = 'dv/dt = 0/ms : 1')

    # create two synapses between the same neurons
    synapses = Synapses(g0, g1, model = 'w:1', pre = 'v+=w')
    synapses[0,0] = 3
    synapses.w[0,0,0] = 1
    synapses.w[0,0,1] = 2
    synapses.w[0,0,2] = 3

    # check the first/last behavior
    w_dense = synapses.w.to_matrix(multiple_synapses = 'sum')
    assert w_dense[0,0] == 6
    w_dense = synapses.w.to_matrix(multiple_synapses = 'max')
    assert w_dense[0,0] == 3
    w_dense = synapses.w.to_matrix(multiple_synapses = 'min')
    assert w_dense[0,0] == 1
    

def test_min_max_sum():
    # prepare neurongroups for the creation of a synapse object
    N0 = 10
    g0 = NeuronGroup(N0, model = 'dv/dt = 0/ms : 1')
    N1 = 10
    g1 = NeuronGroup(N1, model = 'dv/dt = 0/ms : 1')

    # create two synapses between the same neurons
    synapses = Synapses(g0, g1, model = 'w:1', pre = 'v+=w')
    synapses[0,0] = 3
    synapses.w[0,0,0] = 1
    synapses.w[0,0,1] = 2
    synapses.w[0,0,2] = 3

    # check the first/last behavior
    w_dense = synapses.w.to_matrix(multiple_synapses = 'sum')
    assert w_dense[0,0] == 6
    w_dense = synapses.w.to_matrix(multiple_synapses = 'max')
    assert w_dense[0,0] == 3
    w_dense = synapses.w.to_matrix(multiple_synapses = 'min')
    assert w_dense[0,0] == 1
    
if __name__ == '__main__':
    test_first_last()
    test_min_max_sum()
    print "done"
