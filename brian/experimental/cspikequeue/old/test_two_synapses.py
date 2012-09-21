'''
One synapse within several possibilities.
Synapse from 0->2,3.
'''
from brian import *


def test(use_cspikequeue):
    reinit_default_clock()
    clear(all = True)

    P=SpikeGeneratorGroup(3, [(0, 5*ms), (2, 10*ms), (1, 15*ms)])
    Q=NeuronGroup(4,model='v:1')
    S=Synapses(P,Q,model='w:1',pre='v+=w', max_delay = 4*ms)

    S[0,1]=True
    S.w[0,1]=1.
    S[0,2]=True
    S.w[0,2]=2.
    S[0,3]=True
    S.w[0,3]=3.

    S[1,0] = True
    S.w[1,0]= 4.

    S.delay[0,:]=np.array([0*ms, 2*ms, 4*ms])

    M=StateMonitor(Q,'v',record=True)
    Ms=SpikeMonitor(P,record=True)
    run(40*ms)
    return M, Ms, S

def plot_mons(M, Ms):
    for i in range(4):
        plot(M.times/ms,M[i],'k')
    for i in range(3):
        plot(Ms.spiketimes[i]/ms,-0.01*np.ones(len(Ms.spiketimes[i])),'ok')
    ylim(-.5, 5)

if __name__ == '__main__':

    subplot(212)
    M, Ms, S = test(True)
    plot_mons(M, Ms)
    xlabel('With Cspikequeue')
    print 'With'
    print S.queues[0]

    subplot(211)
    M, Ms, S = test(False)
    plot_mons(M, Ms)
    xlabel('Without Cspikequeue')
    print 'Without'
    print S.queues[0]
    print 'Done'

    show()
#1079574528          
