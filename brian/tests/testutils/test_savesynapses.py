'''
This file is a test for the save/load connectivity options of the Synapses class.
See the doc to the save_connectivity method of Synapses for further details on what it does.
This test covers the use case as described in the docstring
'''
from brian.synapses import *
from brian.neurongroup import *
import numpy as np
import cStringIO
from nose.tools import *

def test_save_load_builtin():
    # prepare neurongroups for the creation of a synapse object
    N0 = 10
    g0 = NeuronGroup(N0, model = 'dv/dt = 0/ms : 1')
    N1 = 20
    g1 = NeuronGroup(N1, model = 'dv/dt = 0/ms : 1')

    # create synapses
    synapses = Synapses(g0, g1, model = 'w:1', pre = 'v+=w')
    synapses.connect_random(g0, g1, sparseness = 0.2)
    synapses.w = np.arange(len(synapses))
    
    # use a string io for temporary file
    # this is (almost) unnecessarily complicated, 
    # it's just so that we don't create a file in the local directory
    f = cStringIO.StringIO()

    # save w and the connectivity
    w_before_save = synapses.w[:,:]
    synapses.save_connectivity(f) # SAVE


    # Now, recreate the synapses object
    synapses_after_save = Synapses(g0, g1, model = 'z:1', pre = 'v+=z')    
    # same remark as above
    rf = cStringIO.StringIO(f.getvalue())
    synapses_after_save.load_connectivity(rf) # LOAD
    
    w_after_save = synapses.w[:,:]

    assert (w_after_save == w_before_save).all()
    assert len(synapses) == len(synapses_after_save)
    
if __name__ == '__main__':
    test_save_load_builtin()
