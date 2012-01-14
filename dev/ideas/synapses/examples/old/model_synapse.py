'''
In this script I test the basic Synapses behavior:
synapses = syn.Synapses(gin, gout, model = 'w : 1', pre = 'v += w')
'''

from brian import *
import dev.ideas.synapses.synapses as syn
reload(syn)
log_level_debug()

reinit_default_clock()

if True:
    spikes = [(0, 10*ms), (1, 10*ms), (1, 20*ms), (2, 30*ms)]
    gin = SpikeGeneratorGroup(3, spikes)
    gout = NeuronGroup(2, model = 'dv/dt = -v/(5*ms) : 1')

    eqs = '''
w : 1
u : 1
z = u + v : 1
dx/dt = -x/(10*ms) : 1
'''
    
    pre = '''
v += w ; x += w
'''

    # init synapses
    synapses = syn.Synapses(gin, gout, model = eqs, pre = pre)
    
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
