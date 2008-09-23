# TODO: Change name of SongAbbottSTDP
# TODO: add simpler syntax for spikefn_pre and spikefn_post, e.g. by strings
# TODO: event based update of STDP variables?

from brian import *

__all__ = ['STDP', 'SongAbbottSTDP']

class STDPFunctionConnection(Connection):
    def __init__(self, source, target, stdp, spikefn):
        Connection.__init__(self, source, target)
        self.stdp = stdp
        self.spikefn = spikefn
    def propagate(self, spikes):
        self.spikefn(self.stdp.conn.W, self.stdp.pre, self.stdp.post, spikes)

class STDP(NetworkOperation):
    '''
    Generalised STDP mechanism
    
    Initialised with arguments:
    
    ``conn``
        The :class:`Connection` object which should have STDP. Note that
        at the moment the matrix structure should be ``'dense'``.
    ``eqs_pre``
        The equations defining the presynaptic STDP.
    ``eqs_post``
        The equations defining the postsynaptic STDP.
    ``spikefn_pre``
        The spike function ``f(W, pre, post, spikes)`` used to determine
        STDP behaviour triggered on presynaptic spikes. The arguments
        are ``W`` the connection matrix, ``pre`` and ``post`` are the
        automatically generated ``NeuronGroup`` objects containing
        the variables defined in ``eqs_pre`` and ``eqs_post``, and
        ``spikes`` is the list of spikes.
    ``spikefn_post``
        The spike function for postsynaptic spikes, same arguments
        as ``spikefn_pre``.
    
    The equations can be any set of differential and other equations
    which are updated as for a normal :class:`NeuronGroup`. The
    equations for the presynaptic side will correspond to neurons on
    the presynaptic side, and likewise for the postsynaptic side.
    Whenever the pre or postsynaptic groups fire spikes, these are
    passed on to the corresponding spike functions.
    
    **Example**
    
    Song, Miller and Abbott (2000)'s STDP rule can be implemented for a
    :class:`Connection` ``C`` as follows:
    
        def f_pre(W, pre, post, spikes):
            for i in spikes:
                W[i,:] = clip(W[i,:]+post.A_, 0, gmax)
            pre.A_[spikes] += dA_pre
        def f_post(W, pre, post, spikes):
            for i in spikes:
                W[:,i] = clip(W[:,i]+pre.A_, 0, gmax)
            post.A_[spikes] += dA_post
        stdp = STDP(C,
                    'dA/dt=-A/tau_pre : 1',
                    'dA/dt=-A/tau_post : 1',
                    f_pre, f_post)
    
    Note though that you can more simply use the :class:`SongAbbottSTDP`
    object to do this.
    
    **Implementation notes**
    
    :class:`STDP` is derived from :class:`NetworkOperation` just so that
    it is automatically found by Brian's "magic" functions. It
    automatically generates two :class:`NeuronGroup` objects ``pre``
    and ``post`` corresponding to the pre and postsynaptic neurons
    using the given equations. It also automatically generates two
    new :class:`Connection` objects which pass the spikes from the
    pre and postsynaptic neurons to the spike functions. These objects
    are then added to the returned object's ``contained_objects``
    attribute so that they are automatically added to the :class:`Network`
    object when the returned object is.
    '''
    def __init__(self, conn, eqs_pre, eqs_post, spikefn_pre, spikefn_post):
        if isinstance(eqs_pre, str):
            eqs_pre = Equations(eqs_pre, level=1)
        if isinstance(eqs_post, str):
            eqs_post = Equations(eqs_post, level=1)
        NetworkOperation.__init__(self, lambda:None)
        self.conn = conn
        self.source = conn.source
        self.target = conn.target
        self.pre = NeuronGroup(len(self.source), model=eqs_pre, clock=self.source.clock)
        self.post = NeuronGroup(len(self.target), model=eqs_post, clock=self.target.clock)
        self.pre_conn = STDPFunctionConnection(self.source, self.pre, self, spikefn_pre)
        self.post_conn = STDPFunctionConnection(self.target, self.post, self, spikefn_post)
        self.contained_objects = [self.pre, self.post, self.pre_conn, self.post_conn]
    def __call__(self):
        pass

class SongAbbottSTDP(STDP):
    '''
    STDP rule from Song, Miller and Abbott (2000)
    
    Initialised with arguments:
    
    ``conn``
        The :class:`Connection` object which should have STDP. Note that
        at the moment the matrix structure should be ``'dense'``.
    ``gmax``
        The maximum conductance, in the units of ``conn``'s weights.
    ``tau_pre``
        The STDP rule time constant for pre-before-post spikes.
    ``tau_post``
        The STDP rule time constant for post-before-pre spikes.
    ``dA_pre``
        The STDP rule magnitude for pre-before-post spikes.
    ``dA_post``
        The STDP rule magnitude for post-before-pre spikes.
    
    Here, a presynaptic spike before a postsynaptic spike leads to
    an increase in synaptic weight of dA_pre exp(t/tau_pre) for
    spikes separated by t, and a post-before-pre leads to an increase
    of dA_post exp(t/tau_post) - so note that dA_post should be
    negative for excitatory STDP.
    '''
    def __init__(self, conn, gmax, tau_pre, tau_post, dA_pre, dA_post):
        def f_pre(W, pre, post, spikes):
            for i in spikes:
                W[i,:] = clip(W[i,:]+postA, 0, gmax)
            preA[spikes] += dA_pre
        def f_post(W, pre, post, spikes):
            for i in spikes:
                W[:,i] = clip(W[:,i]+preA, 0, gmax)
            postA[spikes] += dA_post
        STDP.__init__(self, conn,
                      'dA/dt=-A/tau_pre : 1',
                      'dA/dt=-A/tau_post : 1',
                      f_pre, f_post)
        preA = self.pre.A_
        postA = self.post.A_

if __name__=='__main__':
    
    def f():
        
        from time import time
        
        set_global_preferences(useweave=True)
        
        taum=20*ms
        tau_post=20*ms
        tau_pre=20*ms
        Ee=0*mV
        vt=-54*mV
        vr=-60*mV
        El=-70*mV
        taue=5*ms
        gmax=0.015
        dA_pre=gmax*.005
        dA_post=-dA_pre*1.05
        
        eqs_neurons='''
        dv/dt=(ge*(Ee-v)+El-v)/taum : volt
        dge/dt=-ge/taue : 1
        '''
    
        G1 = PoissonGroup(1000, rates=10*Hz)
        G2 = NeuronGroup(1,model=eqs_neurons,threshold=vt,reset=vr)
#        C = Connection(G1, G2, 'ge', structure='dense')
#        C.connect(G1,G2,rand(len(G1),len(G2))*gmax)
        C = Connection(G1, G2, 'ge')
        C.connect_random(G1,G2,0.9,weight=lambda i,j:rand()*gmax)
        
#        def f_pre(W, pre, post, spikes):
#            for i in spikes:
#                W[i,:] = clip(W[i,:]+post.A_,0,gmax)
#            pre.A_[spikes] += dA_pre
#        def f_post(W, pre, post, spikes):
#            for i in spikes:
#                W[:,i]=clip(W[:,i]+pre.A_,0,gmax)
#            post.A_[spikes] += dA_post
#            
#        stdp = STDP(C,
#                    'dA/dt=-A/tau_pre : 1',
#                    'dA/dt=-A/tau_post : 1',
#                    f_pre, f_post )
        
        stdp = SongAbbottSTDP(C, gmax=gmax,
                        tau_pre=tau_pre, tau_post=tau_post,
                        dA_pre=dA_pre, dA_post=dA_post)
        
        rate = PopulationRateMonitor(G2)
        
        G2.v = vr
        
        import stdp_sparse
        C.W = stdp_sparse.SparseSTDPConnectionMatrix(C.W)
        
        start_time=time()
        run(20*second)
        print "Simulation time:",time()-start_time
        
        subplot(211)
        plot(rate.times/ms,rate.smooth_rate(500*ms))
        subplot(212)
        #plot(C.W.squeeze(),'.')
        plot(C.W.alldata,'.')
        show()

    f()