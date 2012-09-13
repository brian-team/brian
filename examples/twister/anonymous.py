#!/usr/bin/env python
'''
Anonymous entry for the 2012 Brian twister.
'''
'''
My contribution to the brian twister!

I meant to give it more thought, but I forgot about the deadline!
'''
from brian import *
from brian.hears import *
import pygame

_mixer_status = [-1,-1]
class SoundMonitor(SpikeMonitor):
    """
    Listen to you networks!
    
    Plays pure tones whenever a neuron spikes, frequency is set according to the neuron number.
    """
    def __init__(self, source, record=False, delay=0, 
                 frange = (100.*Hz, 5000.*Hz),
                 duration = 50*ms,
                 samplerate = 44100*Hz):
        super(SoundMonitor, self).__init__(source, record = record, delay = delay)

        self.samplerate = samplerate
        self.nsamples = np.rint(duration * samplerate)

        p = linspace(0, 1, len(source)).reshape((1, len(source)))
        p = np.tile(p, (self.nsamples, 1))
        freqs = frange[0] * p + (1-p) * frange[1]
        del p

        times = linspace(0*ms, duration, self.nsamples).reshape((self.nsamples, 1))
        times = np.tile(times, (1, len(source)))
        
        self.sounds = np.sin(2 * np.pi * freqs * times) 
        self._init_mixer()


    def propagate(self, spikes):
        if len(spikes):
            data = np.sum(self.sounds[:,spikes], axis = 1)
            x = array((2 ** 15 - 1) * clip(data/amax(data), -1, 1), dtype=int16)
            x.shape = x.size
            # Make sure pygame receives an array in C-order
            x = pygame.sndarray.make_sound(np.ascontiguousarray(x))
            x.play()

    def _init_mixer(self):
        global _mixer_status
        if _mixer_status==[-1,-1] or _mixer_status[0]!=1 or _mixer_status != self.samplerate:
            pygame.mixer.quit()
            pygame.mixer.init(int(self.samplerate), -16, 1)
            _mixer_status=[1,self.samplerate]


def test_cuba():
    # The CUBA example with sound!
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV

    eqs = Equations('''
    dv/dt  = (ge+gi-(v-El))/taum : volt
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    ''')

    P = NeuronGroup(4000, model=eqs, threshold=Vt, reset=Vr, refractory=5 * ms)
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV

    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    Ce = Connection(Pe, P, 'ge', weight=we, sparseness=0.5)
    Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=0.5)
    P.v = Vr + rand(len(P)) * (Vt - Vr)

    # Record the number of spikes
    M = SoundMonitor(P)
    run(10 * second)

def test_synfire():
    from brian import *
    # Neuron model parameters
    Vr = -70 * mV
    Vt = -55 * mV
    taum = 10 * ms
    taupsp = 0.325 * ms
    weight = 4.86 * mV
    # Neuron model
    eqs = Equations('''
    dV/dt=(-(V-Vr)+x)*(1./taum) : volt
    dx/dt=(-x+y)*(1./taupsp) : volt
    dy/dt=-y*(1./taupsp)+25.27*mV/ms+\
        (39.24*mV/ms**0.5)*xi : volt
    ''')
    # Neuron groups
    P = NeuronGroup(N=1000, model=eqs,
        threshold=Vt, reset=Vr, refractory=1 * ms)
    Pinput = PulsePacket(t=50 * ms, n=85, sigma=1 * ms)
    # The network structure
    Pgp = [ P.subgroup(100) for i in range(10)]
    C = Connection(P, P, 'y')
    for i in range(9):
        C.connect_full(Pgp[i], Pgp[i + 1], weight)
    Cinput = Connection(Pinput, Pgp[0], 'y')
    Cinput.connect_full(weight=weight)

    monitor = SoundMonitor(P)

    # Setup the network, and run it
    P.V = Vr + rand(len(P)) * (Vt - Vr)
    run(1*second)
    # Plot result

    show()


if __name__ == '__main__':
    test_synfire()

