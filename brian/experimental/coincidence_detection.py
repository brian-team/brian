from brian import *
from numpy import *

__all__ = ['CoincidenceDetectorGroup']


class CoincidenceDetectorConnection(Connection):
    def propagate(self, spikes):
        self.target.LSST[spikes] = self.source.clock._t


class CoincidenceDetectorStateUpdater(LazyStateUpdater):
    def __init__(self, allow_multiple, ordered):
        self.allow_multiple = allow_multiple
        self.ordered = ordered
        LazyStateUpdater.__init__(self)

    def __call__(self, P):
        lsst = P.LSST
        i0 = P.indices[0]
        i1 = P.indices[1]
        lssti0 = lsst[i0]
        lssti1 = lsst[i1]
        firing = abs(lssti0 - lssti1) < P.delta
        if self.ordered:
            firing = logical_and(firing, lssti0 > lssti1)
        firing = logical_and(firing, lssti0 > P.lastfiring)
        firing = logical_and(firing, lssti1 > P.lastfiring)
        P._S[0][firing] = 1
        if self.allow_multiple:
            # this allows each spike to be used for multiple coincidences
            P.lastfiring[firing] = where(lssti0 > lssti1, lssti0, lssti1) - P.clock._dt * 0.5
        else:
            # this allows each spike to be used for multiple coincidences
            P.lastfiring[firing] = P.clock._t - P.clock._dt * 0.5


class CoincidenceDetectorGroup(NeuronGroup):
    '''
    Perfect pairwise coincidence detector group
    
    Initialisation arguments:
    
    ``source``
        The source group
    ``N``
        The number of coincidence detector neurons
    ``delta``
        The maximum time interval to count as a coincidence
    ``allow_multiple``
        Whether or not a spike can be used for multiple coincidences
        for a given coincidence detector. Suppose neuron 0 fires at
        time 1ms and neuron 1 fires at times 2ms and 3ms with window
        5ms say. If ``allow_multiple=True`` then the coincidence
        detector will fire at times 2ms and 3ms, if ``allow_multiple=False``
        then it will only fire at time 2ms.
    ``ordered``
        If ``False`` and s, t are the spike times then ``abs(s-t)<delta``
        is the coincidence condition, if ``True`` then ``s>t`` must also
        hold.
    
    .. method: set_indices
    
        Takes two arguments, ``i0`` and ``i1`` which should be lists or
        arrays of length the number of neurons in the group. Coincidence
        detector neuron ``k`` receives inputs from ``i0[k]`` and ``i1[k]``
        in the source group.
    '''
    def __init__(self, source, N, delta=1 * ms, allow_multiple=False, ordered=False, clock=None, **kwds):
        if clock is None:
            clock = source.clock
        NeuronGroup.__init__(self, N, model=CoincidenceDetectorStateUpdater(allow_multiple, ordered), reset=0, threshold=0.5, clock=clock, **kwds)
        self._S[:] = 0
        self.source = source
        self.conn = CoincidenceDetectorConnection(source, self)
        self.contained_objects = [self.conn]
        self.LSST = zeros(len(source))
        self.lastfiring = ones(N) * clock._dt * 0.5
        if isinstance(delta, Quantity):
            delta = float(delta)
        self.delta = delta
        self.indices = zeros((2, N), dtype=int)

    def set_indices(self, i0, i1):
        self.indices[0] = i0
        self.indices[1] = i1

    def reinit(self):
        self.LSST[:] = 0
        self.lastfiring[:] = self.clock._dt * 0.5

if __name__ == '__main__':
    if 1:
        G = MultipleSpikeGeneratorGroup(
                [
                 [1 * ms, 2 * ms, 2.1 * ms],
                 [1.5 * ms]
                ])
        H = CoincidenceDetectorGroup(G, 2, 1 * ms, ordered=True, allow_multiple=True)
        H.indices[0] = [0, 1]
        H.indices[1] = [1, 0]
        MG = SpikeMonitor(G)
        MH = SpikeMonitor(H)
        run(10 * ms)
        print MG.spikes
        print MH.spikes
    if 0:
        G = MultipleSpikeGeneratorGroup(
                [
                 [1 * ms, 2.5 * ms],
                 [1.5 * ms]
                ])
        H = CoincidenceDetectorGroup(G, 2)
        H.indices[0] = [0, 1]
        H.indices[1] = [1, 0]
        MG = SpikeMonitor(G)
        MH = SpikeMonitor(H)
        run(10 * ms)
        print MG.spikes
        print MH.spikes
    if 0:
        G = PoissonGroup(2, 250 * Hz)
        H = CoincidenceDetectorGroup(G, 1)
        H.set_indices([0], [1])
        MG = SpikeMonitor(G)
        MH = SpikeMonitor(H)
        run(1 * second)
        raster_plot(MG, MH)
        show()
