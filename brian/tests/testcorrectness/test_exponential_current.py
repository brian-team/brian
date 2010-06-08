from brian import *

def testexponentialcurrent():
    '''Tests whether an exponential current works as predicted
    
    From Tutorial 2b.
    
    The scheme we implement is the following diffential equations:
    
        | taum dV/dt = -V + ge - gi
        | taue dge/dt = -ge
        | taui dgi/dt = -gi
    
    An excitatory neuron connects to state ge, and an inhibitory neuron connects
    to state gi. When an excitatory spike arrives, ge instantaneously increases,
    then decays exponentially. Consequently, V will initially but continuously
    rise and then fall. Solving these equations, if V(0)=0, ge(0)=g0 corresponding
    to an excitatory spike arriving at time 0, and gi(0)=0 then:
    
        | gi = 0    
        | ge = g0 exp(-t/taue)
        | V = (exp(-t/taum) - exp(-t/taue)) taue g0 / (taum-taue)
    '''
    reinit_default_clock()
    taum = 20 * ms
    taue = 1 * ms
    taui = 10 * ms
    Vt = 10 * mV
    Vr = 0 * mV

    spiketimes = [(0, 0 * ms)]

    G1 = SpikeGeneratorGroup(2, spiketimes)
    G2 = NeuronGroup(N=1, model='''
                   dV/dt = (-V+ge-gi)/taum : volt
                   dge/dt = -ge/taue : volt
                   dgi/dt = -gi/taui : volt
                   ''', threshold=Vt, reset=Vr)
    G2.V = Vr

    C1 = Connection(G1, G2, 'ge')
    C2 = Connection(G1, G2, 'gi')

    C1[0, 0] = 3 * mV
    C2[1, 0] = 3 * mV

    Mv = StateMonitor(G2, 'V', record=True)
    Mge = StateMonitor(G2, 'ge', record=True)
    Mgi = StateMonitor(G2, 'gi', record=True)

    run(100 * ms)

    t = Mv.times
    Vpredicted = (exp(-t / taum) - exp(-t / taue)) * taue * (3 * mV) / (taum - taue)

    Vdiff = abs(Vpredicted - Mv[0])

    assert max(Vdiff) < 0.00001

if __name__ == '__main__':
    testexponentialcurrent()
