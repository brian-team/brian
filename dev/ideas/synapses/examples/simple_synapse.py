'''
In this script I test the basic Synapses behavior:
synapses = syn.Synapses(gin, gout, model = 'w : 1', pre = 'v += w')
'''

from brian import *
import dev.ideas.synapses.synapses as syn
reload(syn)
log_level_debug()

reinit_default_clock()

if False:
    # SIMPLE CASE (no delay)
    spikes = [(0, 10*ms), (1, 10*ms), (1, 20*ms), (2, 30*ms)]
    gin = SpikeGeneratorGroup(3, spikes)
    gout = NeuronGroup(2, model = 'dv/dt = -v/(5*ms) : 1')

    # init synapses
    synapses = syn.Synapses(gin, gout, model = '''w : 1; z : 1''', pre = 'v += w')
    
    # set synapses and weights
    synapses[:, 0] = True
    synapses[1, 1] = True

    synapses.w[:, 0] = 1
    synapses.w[1, 1] = 3
    
    # monitor and run
    Mv = StateMonitor(gout, 
'v', record = True)
    net = Network(gin, gout, synapses, Mv)
    net.run(100*ms)
    
    for i in range(len(gout)):
        plot(Mv.times/ms, Mv[i])
    
    show()

    
if False:
    # DELAYS
    spikes = [(0, 10*ms)]
    gin = SpikeGeneratorGroup(3, spikes)
    gout = NeuronGroup(2, model = 'dv/dt = -v/(5*ms) : 1')

    # init synapses
    synapses = syn.Synapses(gin, gout, model = '''w : 1; z : 1''', pre = 'v += w', max_delay = 3*ms)
    
    # set synapses and weights
    synapses[0, 0] = 10
    synapses.w[0, 0] = 1

    synapses.delay[:, 0] = 5*ms
    synapses.delay[:, 0, 5:] = 2*ms

    print synapses.delay

    # monitor and run
    Mv = StateMonitor(gout, 'v', record = True)
    net = Network(gin, gout, synapses, Mv)
    net.run(100*ms)
    
    for i in range(len(gout)):
        plot(Mv.times/ms, Mv[i])
    
    show()


if True:
    # Testing resizing of the spikequeue
    spikes = [(0, 10*ms)]
    
    spikes += [ (i, 30*ms) for i in range(1,10)]
    gin = SpikeGeneratorGroup(11, spikes)
    gout = NeuronGroup(2, model = 'dv/dt = -v/(5*ms) : 1')

    # init synapses
    synapses = syn.Synapses(gin, gout, model = '''w : 1; z : 1''', pre = 'v += w', max_delay = 3*ms)
    
    # set synapses and weights
    synapses[:, 0] = 10
    synapses.w[:, 0] = 1

    synapses.delay[:, 0] = 0*ms
    synapses.delay[:, 0, 5:] = 2*ms
    print synapses.delay

    # monitor and run
    Mv = StateMonitor(gout, 'v', record = True)
    net = Network(gin, gout, synapses, Mv)
    net.run(100*ms)
    
    for i in range(len(gout)):
        plot(Mv.times/ms, Mv[i])
    
    show()


