from brian import *

def testfromtutorial1c():
    '''Tests a behaviour from Tutorial 1c
    
    Solving the differential equation gives:
    
        V = El + (Vr-El) exp (-t/tau)
    
    Setting V=Vt at time t gives:
    
        t = tau log( (Vr-El) / (Vt-El) )
    
    If the simulator runs for time T, and fires a spike immediately
    at the beginning of the run it will then generate n spikes,
    where:
    
        n = [T/t] + 1
    
    If you have m neurons all doing the same thing, you get nm
    spikes. This calculation with the parameters above gives:
    
        t = 48.0 ms
        n = 21
        nm = 840
    
    As predicted.
    '''
    reinit_default_clock()
    tau = 20 * msecond        # membrane time constant
    Vt = -50 * mvolt          # spike threshold
    Vr = -60 * mvolt          # reset value
    El = -49 * mvolt          # resting potential (same as the reset)
    dV = 'dV/dt = -(V-El)/tau : volt # membrane potential'
    G = NeuronGroup(N=40, model=dV, threshold=Vt, reset=Vr)
    G.V = El
    M = SpikeMonitor(G)
    run(1 * second)
    assert M.nspikes == 840

if __name__ == '__main__':
    testfromtutorial1c()
