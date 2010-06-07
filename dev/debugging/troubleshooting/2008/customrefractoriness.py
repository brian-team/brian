from brian import *

__all__=['SimpleCustomRefractoriness', 'CustomRefractoriness']


class SimpleCustomRefractoriness(Refractoriness):
    '''
    Holds the state variable at the custom reset value for a fixed time after a spike.
    
    **Initialised as:** ::
    
        SimpleCustomRefractoriness(resetfunc[,period=5*ms[,state=0]])
    
    with arguments:
    
    ``resetfun``
        The custom reset function ``resetfun(P, spikes)`` for ``P`` a
        :class:`NeuronGroup` and ``spikes`` a list of neurons that
        fired spikes.
    ``period``
        The length of time to hold at the reset value.
    ``state``
        The name or number of the state variable to reset and hold,
        it is your responsibility to check that this corresponds to
        the custom reset function.
    
    The assumption is that ``resetfun(P, spikes)`` will reset the state
    variable ``state`` on the group ``P`` for the spikes with indices
    ``spikes``. The values assigned by the custom reset function are
    stored by this object, and they are clamped at these values for
    ``period``. This object does not introduce refractoriness for more
    than the one specified variable ``state`` or for spike indices
    other than those in the variable ``spikes`` passed to the custom
    reset function.
    '''

    @check_units(period=second)
    def __init__(self, resetfun, period=5*msecond, state=0):
        self.period=period
        self.resetfun=resetfun
        self.state=state
        self._periods={} # a dictionary mapping group IDs to periods
        self.statevectors={}
        self.lastresetvalues={}

    def __call__(self, P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period=self._periods[id(P)]
        else:
            period=int(self.period/P.clock.dt)+1
            self._periods[id(P)]=period
        V=self.statevectors.get(id(P), None)
        if V is None:
            V=P.state_(self.state)
            self.statevectors[id(P)]=V
        LRV=self.lastresetvalues.get(id(P), None)
        if LRV is None:
            LRV=zeros(len(V))
            self.lastresetvalues[id(P)]=LRV
        lastspikes=P.LS.lastspikes()
        self.resetfun(P, lastspikes)             # call custom reset function 
        LRV[lastspikes]=V[lastspikes]         # store a copy of the custom resetted values
        clampedindices=P.LS[0:period]
        V[clampedindices]=LRV[clampedindices] # clamp at custom resetted values

    def __repr__(self):
        return 'Custom refractory period, '+str(self.period)


class CustomRefractoriness(Refractoriness):
    '''
    Holds the state variable at the custom reset value for a fixed time after a spike.
    
    **Initialised as:** ::
    
        CustomRefractoriness(resetfunc[,period=5*ms[,refracfunc=resetfunc]])
    
    with arguments:
    
    ``resetfunc``
        The custom reset function ``resetfunc(P, spikes)`` for ``P`` a
        :class:`NeuronGroup` and ``spikes`` a list of neurons that
        fired spikes.
    ``refracfunc``
        The custom refractoriness function ``refracfunc(P, indices)`` for ``P`` a
        :class:`NeuronGroup` and ``indices`` a list of neurons that are in
        their refractory periods. In some cases, you can choose not to specify this,
        and it will use the reset function.
    ``period``
        The length of time to hold at the reset value.    
    '''

    @check_units(period=second)
    def __init__(self, resetfun, period=5*msecond, refracfunc=None):
        self.period=period
        self.resetfun=resetfun
        if refracfunc is None:
            refracfunc=resetfun
        self.refracfunc=refracfunc
        self._periods={} # a dictionary mapping group IDs to periods

    def __call__(self, P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period=self._periods[id(P)]
        else:
            period=int(self.period/P.clock.dt)+1
            self._periods[id(P)]=period
        lastspikes=P.LS.lastspikes()
        self.resetfun(P, lastspikes)             # call custom reset function
        clampedindices=P.LS[0:period]
        self.refracfunc(P, clampedindices)

    def __repr__(self):
        return 'Custom refractory period, '+str(self.period)

if __name__=='__main__':
    def f(P, spikes):
        P.V[spikes]=rand(len(spikes))*0.5
    R=SimpleCustomRefractoriness(f)
    G=NeuronGroup(5,
            model='''
            dV/dt = -(V-1.1)/(5*ms) : 1
            ''', reset=R, threshold=1)
    M=StateMonitor(G, 'V', record=True)
    run(1*second)
    for i in range(5):
        plot(M.times, M[i])
    show()
