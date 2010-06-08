from brian import *
import pygame
import matplotlib.cm as cm

__all__ = ['RealtimeConnectionMonitor']


class RealtimeConnectionMonitor(NetworkOperation):
    '''
    Realtime monitoring of weight matrix
    
    Short docs:
    
    ``C``
        Connection to monitor
    ``size``
        Dimensions (width, height) of output window, leave as ``None`` to use
        ``C.W.shape``.
    ``scaling``
        If output window dimensions are different to connection matrix, scaling
        is used, options are ``'fast'`` for no interpolation, or ``'smooth'``
        for (slower) smooth interpolation.
    ``wmin, wmax``
        Minimum and maximum weight matrix values, if left to ``None`` then the
        min/max of the weight matrix at each moment is used (and this scaling
        can change over time).
    ``clock``
        Leave to ``None`` for an update every 100ms.
    ``cmap``
        Colour map to use, black and white by default. Get other values from
        ``matplotlib.cm.*``.
    
    Note that this class uses PyGame and due to a limitation with pygame, there
    can be only one window using it. Other options are being considered.
    '''
    def __init__(self, C, size=None, scaling='fast', wmin=None, wmax=None, clock=None, cmap=cm.gray):
        self.C = C
        self.wmin = wmin
        self.wmax = wmax
        self.cmap = cmap
        if clock is None:
            clock = EventClock(dt=100 * ms)
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        pygame.init()
        if size is None:
            width, height = C.W.shape
            self.scaling = None
        else:
            width, height = size
            self.scaling = scaling
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        self.screen_arr = pygame.surfarray.pixels3d(self.screen)

    def __call__(self):
        W = self.C.W.todense()
        wmin, wmax = self.wmin, self.wmax
        if wmin is None: wmin = amin(W)
        if wmax is None: wmax = amax(W)
        if wmax - wmin < 1e-20: wmax = wmin + 1e-20
        W = self.cmap(clip((W - wmin) / (wmax - wmin), 0, 1), bytes=True)
        if self.scaling is None:
            self.screen_arr[:, :, :] = W[:, :, :3]
        elif self.scaling == 'fast':
            srf = pygame.surfarray.make_surface(W[:, :, :3])
            pygame.transform.scale(srf, (self.width, self.height), self.screen)
        elif self.scaling == 'smooth':
            srf = pygame.surfarray.make_surface(W[:, :, :3])
            pygame.transform.smoothscale(srf, (self.width, self.height), self.screen)
        pygame.display.flip()
        pygame.event.pump()

if __name__ == '__main__':
    Nin = 200
    Nout = 200
    duration = 40 * second

    Fmax = 50 * Hz

    tau = 10 * ms
    taue = 2 * ms
    taui = 5 * ms
    sigma = 0.#0.4
    # Note that factor (tau/taue) makes integral(v(t)) the same when the connection
    # acts on ge as if it acted directly on v.
    eqs = Equations('''
    #dv/dt = -v/tau + sigma*xi/(2*tau)**.5 : 1
    dv/dt = (-v+(tau/taue)*ge-(tau/taui)*gi)/tau + sigma*xi/(2*tau)**.5 : 1
    dge/dt = -ge/taue : 1
    dgi/dt = -gi/taui : 1
    excitatory = ge
    inhibitory = gi
    ''')
    reset = 0
    threshold = 1
    refractory = 0 * ms
    taup = 5 * ms
    taud = 5 * ms
    Ap = .1
    Ad = -Ap * taup / taud * 1.2
    wmax_ff = 0.1
    wmax_rec = wmax_ff
    wmax_inh = wmax_rec

    width = 0.2

    recwidth = 0.2

    Gin = PoissonGroup(Nin)
    Gout = NeuronGroup(Nout, eqs, reset=reset, threshold=threshold, refractory=refractory)
    ff = Connection(Gin, Gout, 'excitatory', structure='dense', weight=rand(Nin, Nout) * wmax_ff)
    for i in xrange(Nin):
        ff[i, :] = (rand(Nout) > .5) * wmax_ff
    rec = Connection(Gout, Gout, 'excitatory')
    for i in xrange(Nout):
        d = abs(float(i) / Nout - linspace(0, 1, Nout))
        d[d > .5] = 1. - d[d > .5]
        dsquared = d ** 2
        prob = exp(-dsquared / (2 * recwidth ** 2))
        prob[i] = -1
        inds = (rand(Nout) < prob).nonzero()[0]
        w = rand(len(inds)) * wmax_rec
        rec[i, inds] = w

    inh = Connection(Gout, Gout, 'inhibitory', sparseness=1, weight=wmax_inh)

    stdp_ff = ExponentialSTDP(ff, taup, taud, Ap, Ad, wmax=wmax_ff)
    stdp_rec = ExponentialSTDP(rec, taup, taud, Ap, Ad, wmax=wmax_rec)

    M = RealtimeConnectionMonitor(ff, size=(500, 500), scaling='smooth',
                                  wmin=0., wmax=wmax_ff,
                                  clock=EventClock(dt=20 * ms),
                                  cmap=cm.jet)

    run(0 * ms)

    @network_operation(clock=EventClock(dt=20 * ms))
    def stimulation():
        Gin.rate = Fmax * exp(-(linspace(0, 1, Nin) - rand())**2 / (2 * width ** 2))

    run(duration, report='stderr')
