'''
Make sure that the Synapses class does the same thing as the Connection class.
'''
from nose import SkipTest

from brian import *

def assert_same_voltage(mon, N):
    '''
    Make sure that the first N and last N of the recorded voltage traces
    at exactly the same.
    '''
    for index in range(N):
        assert (mon[index] == mon[N + index]).all()            

def test_without_delays():
    ''' Compare connection via Connection class and via Synapses class.'''
    
    raise SkipTest()

    reinit_default_clock()
    
    N = 10
    duration = (N + 1) * ms
    spikes = [(i, i * ms) for i in range(N)]
    source = SpikeGeneratorGroup(N, spikes)
    
    # one to one connecting
    targets = NeuronGroup(2 * N, model='dv/dt = -v / (1 * ms) : 1', threshold=10, reset=0)
    target_syn = targets[:N]
    target_conn = targets[N:]
    
    conn = Connection(source, target_conn)
    conn.connect_one_to_one(source, target_conn, weight=1.1)
    
    syn = Synapses(source, target_syn, model='w:1', pre='v+=w')
    syn.connect_one_to_one(source, target_syn)
    syn.w = 1.1
    
    mon = StateMonitor(targets, 'v', record=True)
    run(duration)
    assert_same_voltage(mon, N)


def test_with_constant_delays():
    '''
    Compare connection via Connection class and via Synapses class using
    homogeneous delays.
    '''

    raise SkipTest()
        
    reinit_default_clock()
    
    N = 10
    delay = 5 * ms
    duration = (N + 1) * ms + delay
    spikes = [(i, i * ms) for i in range(N)]
    source = SpikeGeneratorGroup(N, spikes)
    
    # one to one connecting
    targets = NeuronGroup(2 * N, model='dv/dt = -v / (1 * ms) : 1', threshold=10, reset=0)
    target_syn = targets[:N]
    target_conn = targets[N:]
    
    conn = Connection(source, target_conn, delay=delay)
    conn.connect_one_to_one(source, target_conn, weight=1.1)
    
    syn = Synapses(source, target_syn, model='w:1', pre='v+=w', max_delay=delay)    
    syn.connect_one_to_one(source, target_syn)
    syn.w = 1.1
    print syn._pre_delays
    syn.delay = delay
    
    mon = StateMonitor(targets, 'v', record=True)
    run(duration)
    assert_same_voltage(mon, N)
    
def test_with_variable_delays():
    '''
    Compare connection via Connection class and via Synapses class using
    heterogeneous delays.
    '''    

    raise SkipTest()

    reinit_default_clock()
    
    N = 10
    flat_delays = array([i * ms for i in range(N)])
    delays = eye(N) * flat_delays

    duration = (N + 1) * ms + delays.max() * second
    spikes = [(i, i * ms) for i in range(N)]
    source = SpikeGeneratorGroup(N, spikes)
    
    # one to one connecting
    targets = NeuronGroup(2 * N, model='dv/dt = -v / (1 * ms) : 1', threshold=10, reset=0)
    target_syn = targets[:N]
    target_conn = targets[N:]
    
    conn = Connection(source, target_conn, delay=True, max_delay=flat_delays.max())
    conn.connect_one_to_one(source, target_conn, weight=1.1, delay=delays)
    
    syn = Synapses(source, target_syn, model='w:1', pre='v+=w')    
    syn.connect_one_to_one(source, target_syn)
    syn.w = 1.1
    syn.delay = flat_delays
    
    mon = StateMonitor(targets, 'v', record=True)
    run(duration)
    assert_same_voltage(mon, N)


if __name__ == '__main__':
    test_without_delays()
    test_with_constant_delays()
    test_with_variable_delays()