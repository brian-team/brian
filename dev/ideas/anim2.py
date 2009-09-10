from brian import *
import matplotlib.cm as cm

__all__ = ['RealtimeConnectionMonitor']

class RealtimeConnectionMonitor(NetworkOperation):
    '''
    Realtime monitoring of weight matrix
    
    Short docs:
    
    ``C``
        Connection to monitor
    ``clock``
        Leave to ``None`` for an update every 100ms.
        
    Other keyword arguments are passed to imshow, with the following default arguments:
    
    * ``interpolation='nearest'``
    * ``origin='lower left'``
    '''
    def __init__(self, C, clock=None, **kwds):
        self.C = C
        if clock is None:
            clock = EventClock(dt=100*ms)
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        self.keywords = {
            'interpolation':'nearest',
            'origin':'lower left'
            }
        self.keywords.update(kwds)
        ion()
        self.img = imshow(self.C.W.todense(), **self.keywords)

    def __call__(self):
        W = self.C.W.todense()
        self.img.set_data(W)
        draw()

if __name__=='__main__':
    Nin = 200
    Nout = 200
    duration = 40*second
    
    Fmax = 50*Hz
    
    tau = 10*ms
    taue = 2*ms
    taui = 5*ms
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
    refractory = 0*ms
    taup = 5*ms
    taud = 5*ms
    Ap = .1
    Ad = -Ap*taup/taud*1.2
    wmax_ff = 0.1
    wmax_rec = wmax_ff
    wmax_inh = wmax_rec
    
    width = 0.2
    
    recwidth = 0.2
    
    Gin = PoissonGroup(Nin)
    Gout = NeuronGroup(Nout, eqs, reset=reset, threshold=threshold, refractory=refractory)
    ff = Connection(Gin, Gout, 'excitatory', structure='dense', weight=rand(Nin, Nout)*wmax_ff)
    for i in xrange(Nin):
        ff[i, :] = (rand(Nout)>.5)*wmax_ff
    rec = Connection(Gout, Gout, 'excitatory')
    for i in xrange(Nout):
        d = abs(float(i)/Nout-linspace(0,1,Nout))
        d[d>.5] = 1.-d[d>.5]
        dsquared = d**2
        prob = exp(-dsquared/(2*recwidth**2))
        prob[i] = -1
        inds = (rand(Nout)<prob).nonzero()[0]
        w = rand(len(inds))*wmax_rec
        rec[i, inds] = w
    
    inh = Connection(Gout, Gout, 'inhibitory', sparseness=1, weight=wmax_inh)
    
    stdp_ff = ExponentialSTDP(ff, taup, taud, Ap, Ad, wmax=wmax_ff)
    stdp_rec = ExponentialSTDP(rec, taup, taud, Ap, Ad, wmax=wmax_rec)
    
    M = RealtimeConnectionMonitor(ff, vmin=0., vmax=wmax_ff,
                                  clock=EventClock(dt=200*ms))
    
    run(0*ms)
    
    @network_operation(clock=EventClock(dt=20*ms))
    def stimulation():
        Gin.rate = Fmax*exp(-(linspace(0,1,Nin)-rand())**2/(2*width**2))
    
    run(duration, report='stderr')