'''
This test script is meant to test the AERSpikeMonitor and the load_aer functions to do efficient spikes i/o.

Good news: 
- loading/saving AER files works

Bad news:
- stacking of spikes makes this test fail if dt is too big. In this case in the SpikeGeneratorGroup some spikes are discarded.

'''
import time
from brian import *
from brian.tools.io import *

N = 1000

def do_spikeio_test(test = 'save', dt = .1*ms):
    '''
    This test function can test two aspects of spike io.
    
    ``test = 'save'`` tests generating a lot of Poisson spikes and saving them. It is then reloaded and the number of spikes is checked for.

    ``test = 'reload'`` tests loading the spikes and putting them directly in a SpikeGeneratorGroup. It's a more thorough test that also tests for the integration with SpikeGeneratorGroup

    '''
    reinit_default_clock()
    clear(all = True)

    defaultclock.dt = dt

    if test == 'save':
        # first generate some spikes
        g = PoissonGroup(N, 200*Hz)
        # with a monitor
        Maer = AERSpikeMonitor(g, './dummy.aedat')
        M = SpikeMonitor(g)
        run(100*ms)
        Maer.close()

        # reload the spikes
        addr, timestamps = load_aer('./dummy.aedat')

        # compare the recorded spikes number etc...
        if len(addr) == M.nspikes:
            print 'Saving is a success'
        else:
            print 'Saving... failed'
            print 'addr, M.spikes', len(addr),M.nspikes


    elif test == 'reload':
        addr, timestamps = load_aer('./dummy.aedat')

        # check interface with SpikeGeneratorGroup
        group = SpikeGeneratorGroup(N, (addr, timestamps))

        newM = SpikeMonitor(group, record = True)

        run(100*ms)

        if len(addr) == newM.nspikes:
            print 'Re-loading is a success'
        else:
            print 'Re-loading... failed'
            print 'addr, M.spikes', len(addr), newM.nspikes


def do_speedtest_saving():
    t0 = time.time()
    g = PoissonGroup(N, 200*Hz)
    # with a monitor
    Maer = AERSpikeMonitor(g, './dummy.aedat')
    run(1000*ms)
    print "AERSpikeMonitor ", time.time()-t0

    reinit_default_clock()
    clear()
    
    t0 = time.time()
    g = PoissonGroup(N, 200*Hz)
    # with a monitor
    Maer = FileSpikeMonitor(g, './dummy.txt')
    run(1000*ms)
    print "FileSpikeMonitor ", time.time()-t0

if __name__ == '__main__':
    print '@dt = 1ms'
    do_spikeio_test(test = 'save', dt = 1*ms)
    do_spikeio_test(test = 'reload', dt = 1*ms)

    print '@dt = 0.1ms'
    do_spikeio_test(test = 'save', dt = .1*ms)
    do_spikeio_test(test = 'reload', dt = .1*ms)

    do_speedtest_saving()
