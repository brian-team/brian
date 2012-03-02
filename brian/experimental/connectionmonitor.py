'''
Ideas for weight monitor:

Should have following features:

* Record from a subset of weights
* Convert to Brian or scipy sparse matrix format
* Callback feature
* Record to file
* Display in GUI
'''
from brian import *
from random import sample
import matplotlib.cm as cm
try:
    import pygame
except ImportError:
    pygame = None

__all__ = ['ConnectionMonitor']

class ConnectionMonitor(NetworkOperation):
    '''
    Monitors synaptic connection weights
    
    Initialisation arguments:
    
    ``C``
        The connection from which to monitor weight values.
    ``matrix='W'``
        Which matrix to record from, 'W' is weights but you can also record
        delays with ``matrix='delay'``.
    ``synapses=None``
        Specify which synapses to record. Can take the following values:
        
        An integer
            Will record this many randomly selected synapses (or less if
            there are less).
        A value between 0 and 1
            Will record this fraction of the synapses, chosen at random.
        ``True``, ``None``
            Records all the synapses
        A list
            The list should consist of pairs ``(i, j)`` to record those
            synapses only.
            
        Note that if an integer or fraction is specified, for a sparse matrix,
        the set of synapses that will be recorded will be chosen at random
        from the nonzero values at the beginning of the run. If you are using
        a :class:`DynamicConnectionMatrix` you might want to specify the
        synapses to record from explicitly.
    ``callback=None``
        A callback function, each time a matrix ``M`` is recorded,
        ``callback(M)`` will be called. This could be used for saving matrices
        to a file, for example, or for runtime analysis. See below for recorded
        matrix format.
    ``store=False``
        Whether or not to keep a copy of the recorded matrices in memory,
        specifically in the ``values`` attribute (see below).
    ``display=False``
        Whether or not to display images at runtime. Can take any of the
        following values:
        
        ``'pygame'``
            Displays the matrix using the pygame module, which must be installed
            if you want to use this. Only one matrix can be recorded at a time
            using pygame.
        
        Note that if ``display`` is not ``False`` it will replace the image
        callback provided (if any).
    ``image_callback=None``
        Each time an image ``I`` is recorded, ``image_callback(I)`` is called.
        See below for image format.
    ``size=None``
        Specifies the dimension of the image. A fairly crude scaling will be used
        if the dimensions of the image are different to those of the matrix.
    ``cmap=jet``
        The colourmap to use. Various values are available in
        ``matplotlib.cm``.
    ``wmin=None, wmax=None``
        The minimum and maximum weights for use with the colourmap. If not
        specified, for each recorded matrix the minimum and maximum values will
        be used.
    ``clock=None``
        By default, matrices will be recorded every 1s of simulated time.
    ``when='end'``
        When the matrices are recorded (see :class:`NetworkOperation`).
        
    **Matrix format**
    
    Matrices are recorded in one of two formats, either a numpy ``ndarray``
    if the matrix to be recorded is dense and all synapses are being recorded,
    or a Brian :class:`SparseMatrix` (which is derived from
    ``scipy.sparse.lil_matrix``) in all other cases. If a subset of synapses
    to record has been selected, the sparse matrix will consist only of those
    synapses.
    
    **Image format**
    
    The images are numpy arrays with shape ``(width, height, 4)`` and 
    ``dtype=uint8``. The last axis is the colour in the format RGBA, with
    values between 0 and 255. The A component is alpha, and will always be
    255.
        
    **Attributes**
    
    ``values``
        If ``store=True`` this consists of a list of pairs ``(t, M)`` giving
        the recorded matrix ``M`` at time ``t``.
    '''
    def __init__(self, C, matrix='W', synapses=None,
                 callback=None, store=False,
                 display=False,
                 image_callback=None, size=None,
                 cmap=cm.jet, wmin=None, wmax=None,
                 clock=None, when='end',
                 ):
        self.C = C
        self.matrix = matrix
        self.synapses = synapses
        self.synapse_set = None
        self.callback = callback
        self.display = display
        self.store = store
        self.image_callback = image_callback
        Wh, Ww = getattr(C, matrix).shape
        if size is None:
            size = Ww, Wh
        w, h = size
        self.size = (h, w)
        self.wmin = wmin
        self.wmax = wmax
        self.cmap = cmap
        if clock is None:
            clock = EventClock(dt=1*second)
        if display=='pygame':
            if pygame is None:
                warnings.warn('Need pygame module to use pygame display mode.')
            else:
                self.screen = pygame.display.set_mode(self.size[::-1])
                self.screen_arr = pygame.surfarray.pixels3d(self.screen)
                self.image_callback = self.pygame_image_callback            
        NetworkOperation.__init__(self, lambda:None, clock=clock, when=when)
        self.reinit()

    def reinit(self):
        self.values = []
        
    def get_matrix(self):
        W = getattr(self.C, self.matrix)
        sparseW = isinstance(W.get_row(0), SparseConnectionVector)
        if not(self.synapses is None or self.synapses is True):
            if self.synapse_set is None:
                # The first time we run we generate the synapse set
                # We don't do this on initialisation because the user
                # might initialise the matrix values before defining the
                # monitor.
                if sparseW:
                    nnz = W.getnnz()
                else:
                    nnz = W.shape[0]*W.shape[1]
                synapses = self.synapses
                if isinstance(synapses, float):
                    synapses = int(synapses*nnz)
                if isinstance(synapses, int):
                    if synapses>=nnz:
                        # recording all synapses, so use the code below
                        self.synapses = None
                    else:
                        # We want to pick a sample of indices at random, so
                        # we pick a subset of [0...nnz-1] and pick out the
                        # corresponding synapses (i,j) if they are arranged
                        # in order
                        indices = sample(xrange(nnz), synapses)
                        indices.sort()
                        indices = array(indices, dtype=int)
                        if not sparseW:
                            # For dense matrices there is a simple conversion
                            # from 1D indices to 2D indices
                            i = indices%W.shape[0]
                            j = indices/W.shape[0]
                            synapses = zip(i, j)
                        else:
                            # For sparse matrices, we need to go through row
                            # by row and pick out the corresponding indices
                            # Note that we don't return a list of pairs
                            # (i,j) in this case, but directly create the
                            # synapse_set as it is more efficient
                            ss = SparseMatrix(W.shape)
                            for i in xrange(W.shape[0]):
                                row = W.get_row(i)
                                I = indices<len(row)
                                ss[i, row.ind[indices[I]]] = 1
                                indices = indices[-I]-len(row)
                            self.synapse_set = ss
                # synapses should now be a list of pairs (i, j)
                if self.synapse_set is None:
                    # synapse_set is a sparse matrix with nonzero elements
                    # in the places we want to record from
                    self.synapse_set = SparseMatrix(W.shape)
                    for u, v in synapses:
                        self.synapse_set[u, v] = 1
            # now we record a new sparse matrix by looping through the rows
            # and copying the data across
            M = SparseMatrix(W.shape)
            for i in xrange(W.shape[0]):
                inds = self.synapse_set.rows[i]
                if len(inds):
                    row = W.get_row(i)
                    # we convert to dense vector because we want to extract
                    # certain indices, and these indices may not exist in
                    # the underlying sparse vector (i.e. they may be zeros)
                    if isinstance(row, SparseConnectionVector):
                        row = row.todense()
                    M[i, inds] = row[inds]
        if self.synapses is None or self.synapses is True:
            # If we're recording everything it's fairly simple, we just
            # make a copy.
            if sparseW:
                M = SparseMatrix(W.shape)
            else:
                M = zeros(W.shape)
            for i in xrange(W.shape[0]):
                row = W.get_row(i)
                if sparseW:
                    M[i, row.ind] = asarray(row)
                else:
                    M[i, :] = row
        return M
    
    def get_image(self, W=None):
        '''
        Returns an image corresponding to the matrix W, scaling down to
        the appropriate size.
        '''
        wmin, wmax = self.wmin, self.wmax
        if W is None:
            W = getattr(self.C, self.matrix)
        h, w = self.size
        Wh, Ww = W.shape
        # We initially generate the image at the maximum size of the matrix,
        # and scale up if the requested size was bigger
        if w>Ww: w = Ww
        if h>Wh: h = Wh
        image = zeros((w, h))
        # We're going to do some crude scaling by adding all the weights and
        # dividing by the number of itmes, for each pixel of the image
        image_numitems = zeros((w, h), dtype=int)
        ind = arange(W.shape[1])
        # We will do histograms for each row, adding up the corresponding
        # weights for indices in these bins
        bins = hstack(((arange(w)*Ww)/w, Ww))
        for i in xrange(W.shape[0]):
            u = h-1-((i*h)//Wh) # target coordinate, scaled
            row = W.get_row(i)
            if isinstance(row, SparseConnectionVector):
                ind = row.ind
            numitems, _ = histogram(ind, bins)
            vals, _ = histogram(ind, bins, weights=row)
            image[:, u] += vals
            image_numitems[:, u] += numitems
        image_numitems[image_numitems==0] = 1
        image = image/image_numitems
        if wmin is None: wmin = amin(image)
        if wmax is None: wmax = amax(image)
        if wmax - wmin < 1e-20: wmax = wmin + 1e-20
        image = self.cmap(clip((image - wmin) / (wmax - wmin), 0, 1), bytes=True)
        # Scale up if the requested image size is larger than the matrix
        ih, iw = self.size
        if ih>h or iw>w:
            yind = (arange(ih)*h)/ih
            xind = (arange(iw)*w)/iw
            xind, yind = meshgrid(xind, yind)
            image = image[xind.T, yind.T, :]
        return image
        
    def pygame_image_callback(self, image):
        self.screen_arr[:, :, :3] = image[:, :, :3]
        pygame.display.flip()
        pygame.event.pump()
        
    def __call__(self):
        W = self.get_matrix()
        if self.callback is not None:
            self.callback(W)
        if self.store:
            self.values.append((self.clock.t, W))
        if self.image_callback is not None:
            self.image_callback(self.get_image())

if __name__=='__main__':
    if 0:
        N = 1000
        taum = 10 * ms
        tau_pre = 20 * ms
        tau_post = tau_pre
        Ee = 0 * mV
        vt = -54 * mV
        vr = -60 * mV
        El = -74 * mV
        taue = 5 * ms
        F = 15 * Hz
        gmax = .01
        dA_pre = .01
        dA_post = -dA_pre * tau_pre / tau_post * 1.05
        
        eqs_neurons = '''
        dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
        dge/dt=-ge/taue : 1
        '''
        
        input = PoissonGroup(N, rates=F)
        neurons = NeuronGroup(1, model=eqs_neurons, threshold=vt, reset=vr)
        synapses = Connection(input, neurons, 'ge', weight=rand(len(input), len(neurons)) * gmax)
        neurons.v = vr
        
        #stdp=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,dA_post,wmax=gmax)
        ## Explicit STDP rule
        eqs_stdp = '''
        dA_pre/dt=-A_pre/tau_pre : 1
        dA_post/dt=-A_post/tau_post : 1
        '''
        dA_post *= gmax
        dA_pre *= gmax
        stdp = STDP(synapses, eqs=eqs_stdp, pre='A_pre+=dA_pre;w+=A_post',
                  post='A_post+=dA_post;w+=A_pre', wmax=gmax)
        
        rate = PopulationRateMonitor(neurons)
        
        M = ConnectionMonitor(synapses, store=True, synapses=200)
        
        run(100 * second, report='text')
        
        Z = zeros((100, 1000))
        
        for i, (_, m) in enumerate(M.values):
            #print m[:, 0].__class__
            Z[i, :] = m.todense().flatten()
            
        imshow(Z, aspect='auto')
        show()
    else:
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
        ff = Connection(Gin, Gout, 'excitatory', structure='dense')
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
    
        M = ConnectionMonitor(ff, display='pygame',
                              #size=(500,500),
                              #synapses=10000,
                              )
        
        run(0 * ms)
        @network_operation(clock=EventClock(dt=20 * ms))
        def stimulation():
            Gin.rate = Fmax * exp(-(linspace(0, 1, Nin) - rand())**2 / (2 * width ** 2))
    
        run(40*second, report='stderr')

        if 0:
            for i, (_, m) in enumerate(M.values):
                if i%16==0:
                    figure()
                subplot(4, 4, i%16+1)
                imshow(array(m.todense()), aspect='auto',
                       #interpolation='nearest'
                       )
            show()