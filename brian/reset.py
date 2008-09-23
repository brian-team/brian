# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
'''
Reset mechanisms
'''

__all__=['Reset','VariableReset','Refractoriness','NoReset','FunReset',
         'CustomRefractoriness', 'SimpleCustomRefractoriness']

from numpy import where, zeros
from units import *
from clock import *

def _define_and_test_interface(self):
    """
    :class:`Reset`
    ~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = Reset(resetvalue=0*mvolt, state=0)

    After a neuron from a group with this reset fires, it
    will set the specified state variable to the given value.
    State variable 0 is customarily the membrane voltage,
    but this isn't required. 
    
    :class:`FunReset`
    ~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = FunReset(resetfun)
    
    Where resetfun is a function taking two arguments, the group
    it is acting on, and the indices of the spikes to be reset.
    The following is an example reset function::
    
        def f(P,spikeindices):
            P._S[0,spikeindices]=array([i/10. for i in range(len(spikeindices))])    
    
    :class:`Refractoriness`
    ~~~~~~~~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = Refractoriness(resetvalue=0*mvolt,period=5*msecond,state=0)
    
    After a neuron from a group with this reset fires, the specified state
    variable of the neuron will be set to the specified resetvalue for the
    specified period.
    
    :class:`NoReset`
    ~~~~~~~~~~~~~~~~
    
    Initialised as::
    
        R = NoReset()
        
    Does nothing.
    """
    
    from network import Network, network_operation
    from numpy import array, zeros, ones
    from neurongroup import NeuronGroup
    from stateupdater import LazyStateUpdater, FunStateUpdater
    from threshold import Threshold
    from utils.approximatecomparisons import is_approx_equal
    
    # test that reset works as expected
    # the setup below is that group G starts with state values (1,1,1,1,1,0,0,0,0,0) threshold
    # value 0.5 (which should be initiated for the first 5 neurons) and reset 0.2 so that the
    # final state should be (0.2,0.2,0.2,0.2,0.2,0,0,0,0,0) 
    G = NeuronGroup(10,model=LazyStateUpdater(),reset=Reset(0.2),threshold=Threshold(0.5),init=(0.,))
    G1 = G.subgroup(5)
    G2 = G.subgroup(5)
    G1.state(0)[:] = array([1.]*5)
    G2.state(0)[:] = array([0.]*5)
    net = Network(G)
    net.run(1*msecond)
    self.assert_(all(G1.state(0)<0.21) and all(0.19<G1.state(0)) and all(G2.state(0)<0.01))
    
    # check that function reset works as expected
    def f(P,spikeindices):
        P._S[0,spikeindices]=array([i/10. for i in range(len(spikeindices))])
        P.called_f = True
    G = NeuronGroup(10,model=LazyStateUpdater(),reset=FunReset(f),threshold=Threshold(2.),init=(3.,))
    G.called_f = False
    net = Network(G)
    net.run(1*msecond)
    self.assert_(G.called_f)
    for i,v in enumerate(G.state(0)):
        self.assert_(is_approx_equal(i/10.,v))

    # check that refractoriness works as expected
    # the network below should start at V=15, immediately spike as it is above threshold=1,
    # then should be clamped at V=-.5 until t=1ms at which point it should quickly evolve
    # via the DE to a value near 0 (and certainly between -.5 and 0). We test that the
    # value at t=0.5 is exactly -.5 and the value at t=1.5 is between -0.4 and 0.1 (to
    # avoid floating point problems)
    dV = 'dV/dt=-V/(.1*msecond):1.'
    G = NeuronGroup(1,model=dV,threshold=1.,reset=Refractoriness(-.5,1*msecond))
    G.V = 15.
    net = Network(G)
    net.run(0.5*msecond)
    for v in G.state(0):
        self.assert_(is_approx_equal(v,-.5))
    net.run(1*msecond)
    for v in G.state(0):
        self.assert_(-0.4<v<0.1)
        
    get_default_clock().reinit()
    


class Reset(object):
    '''
    Resets specified state variable to a fixed value
    
    **Initialise as:** ::
    
        R = Reset([resetvalue=0*mvolt[, state=0]])
        
    with arguments:
    
    ``resetvalue``
        The value to reset to.
    ``state``
        The name or number of the state variable to reset.

    This will reset all of the neurons that have just spiked. The
    given state variable of the neuron group will be set to value
    ``resetvalue``.
    '''
    def __init__(self,resetvalue=0*mvolt,state=0):
        self.resetvalue=resetvalue
        self.state = state
        self.statevectors = {}
        
    def __call__(self,P):
        '''
        Clamps membrane potential at reset value.
        '''
        V = self.statevectors.get(id(P),None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        V[P.LS.lastspikes()] = self.resetvalue
        
    def __repr__(self):
        return 'Reset '+str(self.resetvalue)

class VariableReset(Reset):
    '''
    Resets specified state variable to the value of another state variable
    
    Initialised with arguments:
    
    ``resetvaluestate``
        The state variable which contains the value to reset to.
    ``state``
        The name or number of the state variable to reset.

    This will reset all of the neurons that have just spiked. The
    given state variable of the neuron group will be set to
    the value of the state variable ``resetvaluestate``.
    '''
    def __init__(self, resetvaluestate=1, state=0):
        self.resetvaluestate=resetvaluestate
        self.state = state
        self.resetstatevectors = {}
        self.statevectors = {}
        
    def __call__(self,P):
        '''
        Clamps membrane potential at reset value.
        '''
        V = self.statevectors.get(id(P),None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        Vr = self.resetstatevectors.get(id(P),None)
        if Vr is None:
            Vr = P.state_(self.resetvaluestate)
            self.resetstatevectors[id(P)] = Vr
        lastspikes = P.LS.lastspikes()
        V[lastspikes] = Vr[lastspikes]
        
    def __repr__(self):
        return 'VariableReset('+str(self.resetvaluestate)+', '+str(self.state)+')'


class FunReset(Reset):
    '''
    A reset with a user-defined function.
    
    **Initialised as:** ::
    
        FunReset(resetfun)
    
    with argument:
    
    ``resetfun``
        A function ``f(G,spikes)`` where ``G`` is the
        :class:`NeuronGroup` and ``spikes`` is an array of
        the indexes of the neurons to be reset.
    '''
    def __init__(self,resetfun):
        self.resetfun=resetfun
        
    def __call__(self,P):
        self.resetfun(P,P.LS.lastspikes())


class Refractoriness(Reset):
    '''
    Holds the state variable at the reset value for a fixed time after a spike.

    **Initialised as:** ::
    
        Refractoriness([resetvalue=0*mV[,period=5*ms[,state=0]]])
    
    with arguments:
    
    ``resetvalue``
        The value to reset and hold to.
    ``period``
        The length of time to hold at the reset value.
    ``state``
        The name or number of the state variable to reset and hold.
    '''
    @check_units(period=second)
    def __init__(self,resetvalue=0*mvolt,period=5*msecond,state=0):
        #self.period=int(period/guess_clock(clock).dt)
        self.period = period
        self.resetvalue = resetvalue
        self.state = state
        self._periods = {} # a dictionary mapping group IDs to periods
        self.statevectors = {}
        
    def __call__(self,P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period/P.clock.dt)+1
            self._periods[id(P)] = period
        V = self.statevectors.get(id(P),None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        V[P.LS[0:period]] = self.resetvalue
        
    def __repr__(self):
        return 'Refractory period, '+str(self.period)

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
        self.period = period
        self.resetfun = resetfun
        self.state = state
        self._periods = {} # a dictionary mapping group IDs to periods
        self.statevectors = {}
        self.lastresetvalues = {}

    def __call__(self,P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period/P.clock.dt)+1
            self._periods[id(P)] = period
        V = self.statevectors.get(id(P),None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        LRV = self.lastresetvalues.get(id(P),None)
        if LRV is None:
            LRV = zeros(len(V))
            self.lastresetvalues[id(P)] = LRV
        lastspikes = P.LS.lastspikes()
        self.resetfun(P,lastspikes)             # call custom reset function 
        LRV[lastspikes] = V[lastspikes]         # store a copy of the custom resetted values
        clampedindices = P.LS[0:period] 
        V[clampedindices] = LRV[clampedindices] # clamp at custom resetted values
        
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
        self.period = period
        self.resetfun = resetfun
        if refracfunc is None:
            refracfunc = resetfun
        self.refracfunc = refracfunc 
        self._periods = {} # a dictionary mapping group IDs to periods

    def __call__(self,P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period/P.clock.dt)+1
            self._periods[id(P)] = period
        lastspikes = P.LS.lastspikes()
        self.resetfun(P,lastspikes)             # call custom reset function
        clampedindices = P.LS[0:period] 
        self.refracfunc(P,clampedindices) 
        
    def __repr__(self):
        return 'Custom refractory period, '+str(self.period)
    
class NoReset(Reset):
    '''
    Absence of reset mechanism.
    
    **Initialised as:** ::
    
        NoReset()
    '''
    def __init__(self):
        pass
    
    def __call__(self,P):
        pass
    
    def __repr__(self):
        return 'No reset'