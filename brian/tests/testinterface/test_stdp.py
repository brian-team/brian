from brian import *
from brian.tests import repeat_with_global_opts

@repeat_with_global_opts([{'usecstdp': False,  'useweave': False},
                          {'usecstdp': True,  'useweave': True}])
def test_stdp():
    '''
    Test that the basic STDP mechanism works. Does not really test for
    correctness, only asserts that weight increases after the spike.
    ''' 
    reinit()
    # Set up a very simple network that receives an input spike and spikes
    # about 2ms later
    inp = SpikeGeneratorGroup(1, [(0, 1 * ms)])
    Ee, El = 0 * mV, - 80 * mV    
    eqs_neurons = '''
    dv/dt = (ge * (Ee - v) + (El - v)) / (5 * ms) : volt   
    dge/dt = -ge / (5 * ms) : 1
    '''
    G = NeuronGroup(1, model=eqs_neurons, threshold=-50 * mV, reset=El)    
    G.v = -51 * mV      
    con = Connection(inp, G, 'ge', weight=1)
    
    # Set up the STDP
    eqs_stdp = """
                 dA_pre/dt  = -A_pre/(10 * ms)   : 1
                 dA_post/dt = -A_post/(10 * ms) : 1
                 """
    delta_A_pre, delta_A_post = 0.1, 0.1
    stdp = STDP(con, eqs=eqs_stdp, pre='A_pre+=delta_A_pre; w+=A_post',
                post='A_post+=delta_A_post; w+=A_pre')

    net = Network(inp, G, con, stdp)
    net.run(4 * ms)
    
    # Postsynaptic spike came after presynaptic spike: weight should increase
    assert(con.W[0, 0] > 1) 

if __name__ == '__main__':
    test_stdp()