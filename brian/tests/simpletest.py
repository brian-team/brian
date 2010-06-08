from brian import *

__all__ = ['brian_sample_run']

def brian_sample_run():
    '''
    A simple Brian script for users to test they have installed Brian correctly.
    '''
    reinit_default_clock()
    clear(True)
    eqs = '''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    P = NeuronGroup(4000, eqs, threshold= -50 * mV, reset= -60 * mV)
    P.v = -60 * mV + 10 * mV * rand(len(P))
    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    Ce = Connection(Pe, P, 'ge')
    Ci = Connection(Pi, P, 'gi')
    Ce.connect_random(Pe, P, 0.02, weight=1.4 * mV)
    Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)
    M = SpikeMonitor(P)
    run(1000 * ms)
    raster_plot(M)
    print 'Brian sample run finished OK!'
    show()

if __name__ == '__main__':
    brian_sample_run()
