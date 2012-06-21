import sys
from StringIO import StringIO

from numpy.testing.utils import assert_raises

from brian import *
from brian.utils.progressreporting import ProgressReporter



def test_progressreporting():
    '''
    Tests whether calling for progress reporting works -- the actual output is
    not checked, only that something is printed to the correct output.
    '''
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    
    G = NeuronGroup(1, model=LazyStateUpdater())
    net = Network(G)
    # Tests various ways to get a (textual) progress report
    string_output = StringIO()
    sys.stdout = string_output
    net.run(defaultclock.dt, report='text')
    assert(len(string_output.getvalue()) > 0)
        
    net.run(defaultclock.dt, report='print')
    string_output = StringIO()
    sys.stdout = string_output    
    net.run(defaultclock.dt, report='stdout')
    assert(len(string_output.getvalue()) > 0)
        
    string_output = StringIO()
    sys.stdout = string_output
    string_err = StringIO()
    sys.stderr = string_err
    net.run(defaultclock.dt, report='stderr')    
    assert(len(string_output.getvalue()) == 0 and len(string_err.getvalue()) > 0)
    
    net.run(defaultclock.dt, report=string_output)
    assert(len(string_output.getvalue()) > 0)
    net.run(defaultclock.dt, report='text', report_period=5*second)
    
    #explicitely construct a ProgressReporter    
    string_output = StringIO()
    sys.stdout = string_output
    reporter = ProgressReporter('text')
    reporter.start()
    net.run(defaultclock.dt, report=reporter)
    assert(len(string_output.getvalue()) > 0)
    
    string_output = StringIO()
    sys.stdout = string_output
    string_err = StringIO()
    sys.stderr = string_err
    reporter = ProgressReporter('stderr')
    net.run(defaultclock.dt, report=reporter)    
    assert(len(string_output.getvalue()) == 0 and len(string_err.getvalue()) > 0)
    
    # use MagicNetwork implicitly
    string_output = StringIO()
    sys.stdout = string_output
    run(defaultclock.dt, report='text', report_period=5*second)
    assert(len(string_output.getvalue()) > 0)
    
    sys.stdout = real_stdout
    sys.stderr = real_stderr

def test_network_generation():
    '''
    Tests various ways of creating a network
    '''
    G = NeuronGroup(42, model=LazyStateUpdater())
    net = Network(G)
    assert(len(net) == 42)
    
    net = Network()
    net.add(G)
    assert(len(net) == 42)
    
    #try adding a group twice
    net.add(G)
    assert(len(net) == 42)
    
    #try adding something that should not be possible to add
    assert_raises(TypeError, lambda: net.add(1 * msecond))    
    
    net = Network()
    # the network's call function adds an object and returns it
    assert(net(G) is G)
    assert(len(net) == 42)
    
    net = MagicNetwork()
    assert(len(net) == 42)
    
    net = MagicNetwork(verbose=True)
    assert(len(net) == 42)
    
    # Test that 'forgetting' a group works
    forget(G)
    net = MagicNetwork(verbose=True)
    assert(len(net) == 0)
    
    # Test that 'recalling' works
    recall(G)
    net = MagicNetwork(verbose=True)
    assert(len(net) == 42)
    
    # Repeat the test with a list of objects
    # Test that 'forgetting' a group works
    forget([G])
    net = MagicNetwork(verbose=True)
    assert(len(net) == 0)
    
    # Test that 'recalling' works
    recall([G])
    net = MagicNetwork(verbose=True)
    assert(len(net) == 42)

def test_reinit():
    '''
    Test whether reinitializing works correctly and only resets what it is 
    supposed to reset.
    '''
    # setup a simple network
    eqs = '''dv/dt = -v / (1 * ms) : 1 # a variable
        p : 1 # a parameter
        '''
    G = NeuronGroup(1, model=eqs)
    G.v = 0.1
    G.p = 1
    
    net = Network(G)
    net.run(defaultclock.dt)
    assert(G.v > 0 and G.p == 1)
    
    #reinit(False) should only reset the clock
    net.reinit(False)
    assert(defaultclock.t == 0)
    
    #reinit(True) should reset everyting
    net.run(defaultclock.dt)
    net.reinit(True)
    assert(defaultclock.t == 0)
    assert(G.v == 0 and G.p == 0)
    

def test_network_clocks():
    '''
    Test whether setting and checking clocks works
    '''
    
    # Two groups using the same (default) clock
    G1 = NeuronGroup(42, model=LazyStateUpdater())
    G2 = NeuronGroup(42, model=LazyStateUpdater())
    net = Network(G1, G2)
    assert(net.same_clocks())    
    
    # Two groups using the same clocks
    clock = Clock(dt=0.333*ms)
    G1 = NeuronGroup(42, model=LazyStateUpdater(), clock=clock)
    G2 = NeuronGroup(42, model=LazyStateUpdater(), clock=clock)
    net = Network(G1, G2)
    assert(net.same_clocks())    
    
    # Two groups using different clocks
    clock2 = Clock(dt=0.5*ms)
    G1 = NeuronGroup(42, model=LazyStateUpdater(), clock=clock)
    G2 = NeuronGroup(42, model=LazyStateUpdater(), clock=clock2)
    net = Network(G1, G2)
    assert(not net.same_clocks())

def test_network_operation():
    reinit_default_clock()
    G = NeuronGroup(1, model='dv/dt = -v / (1 * ms) : 1')
    G.v = 1
    @network_operation(when='end')
    def keep_voltage(clock):
        #reset voltage to 1 after the update step
        G.v = 1
    mon = StateMonitor(G, 'v', record=True)
    net = Network(G, keep_voltage, mon)
    net.run(5 * defaultclock.dt)
    assert(all(mon[0] == 1.0))
    
    
    #test a dummy NetworkOperation that does nothing
    mon.reinit()
    
    @network_operation()
    def do_nothing(clock):
        pass
    net = Network(G, do_nothing, mon)
    net.run(5 * defaultclock.dt)
    # test that there is some decay
    assert(all(mon[0] < 1.0))

    
if __name__ == '__main__':
    test_progressreporting()
    test_network_generation()
    test_network_clocks()
    test_network_operation()
    test_reinit()