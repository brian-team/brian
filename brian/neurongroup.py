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
Neuron groups
'''

__all__=['NeuronGroup','PoissonGroup']

from numpy import *
from scipy import rand,linalg,random
import copy
from units import *
from threshold import *
import bisect
from reset import *
from clock import *
from stateupdater import *
from inspection import *
from operator import isSequenceType
import types
from utils.circular import *
import magic
from itertools import count
from equations import *
from globalprefs import *
from quantityarray import *
from neuronmodel import *
import sys
from brian_unit_prefs import bup
import numpy
from base import *
from group import *
from threshold import select_threshold
from reset import select_reset

class NeuronGroup(magic.InstanceTracker, ObjectContainer, Group):
    """Group of neurons
    
    Initialised with arguments:
    
    ``N``
        The number of neurons in the group.
    ``model``
        An object defining the neuron model. It can be a ``Model`` object,
        an :class:`Equations` object, a string defining an :class:`Equations` object,
        a :class:`StateUpdater` object, or a list or tuple of :class:`Equations` and
        strings.
    ``threshold=None``
        A :class:`Threshold` object, a function, a scalar quantity or a string.
        If ``threshold`` is a function with one argument, it will be
        converted to a :class:`SimpleFunThreshold`, otherwise it will be a
        :class:`FunThreshold`. If ``threshold`` is a scalar, then a constant
        single valued threshold with that value will be used. In this case,
        the variable to apply the threshold to will be guessed. If there is
        only one variable, or if you have a variable named one of
        ``V``, ``Vm``, ``v`` or ``vm`` it will be used. If ``threshold`` is a
        string then the appropriate threshold type will be chosen, for example
        you could do ``threshold='V>10*mV'``. The string must be a one line
        string.
    ``reset=None``
        A :class:`Reset` object, a function, a scalar quantity or a string. If it's a
        function, it will be converted to a :class:`FunReset` object. If it's
        a scalar, then a constant single valued reset with that value will
        be used. In this case,
        the variable to apply the reset to will be guessed. If there is
        only one variable, or if you have a variable named one of
        ``V``, ``Vm``, ``v`` or ``vm`` it will be used. If ``reset`` is a
        string it should be a series of expressions which are evaluated for
        each neuron that is resetting. The series of expressions can be
        multiline or separated by a semicolon. For example,
        ``reset=`Vt+=5*mV; V=Vt'``. Statements involving ``if`` constructions
        will often not work because the code is automatically vectorised.
        For such constructions, use a function instead of a string.
    ``refractory=0*ms``
        A refractory period, used in combination with the ``reset`` value
        if it is a scalar.
    ``clock``
        A clock to use for scheduling this :class:`NeuronGroup`, if omitted the
        default clock will be used.
    ``order=1``
        The order to use for nonlinear differential equation solvers.
        TODO: more details.
    ``implicit=False``
        Whether to use an implicit method for solving the differential
        equations. TODO: more details.
    ``max_delay=0*ms``
        The maximum allowable delay (larger values use more memory).
        This doesn't usually need to be specified because Connections will update it.
    ``compile=False``
        Whether or not to attempt to compile the differential equation
        solvers (into Python code). Typically, for best performance, both ``compile``
        and ``freeze`` should be set to ``True`` for nonlinear differential equations.
    ``freeze=False``
        If True, parameters are replaced by their values at the time
        of initialization.
    ``method=None``
        If not None, the integration method is forced. Possible values are
        linear, nonlinear, Euler, exponential_Euler (overrides implicit and order
        keywords).
    ``unit_checking=True``
        Set to ``False`` to bypass unit-checking.
    
    **Methods**
    
    .. method:: subgroup(N)
    
        Returns the next sequential subgroup of ``N`` neurons. See
        the section on subgroups below.

    .. method:: state(var)
                
        Returns the array of values for state
        variable ``var``, with length the number of neurons in the
        group.
    
    .. method:: rest()
    
        Sets the neuron state values at rest for their differential
        equations.

    The following usages are also possible for a group ``G``:
    
    ``G[i:j]``
        Returns the subgroup of neurons from ``i`` to ``j``.
    ``len(G)``
        Returns the number of neurons in ``G``.
    ``G.x``
        For any valid Python variable name ``x`` corresponding to
        a state variable of the the :class:`NeuronGroup`, this
        returns the array of values for the state
        variable ``x``, as for the :meth:`state` method
        above.
    
    **Subgroups**
    
    A subgroup is a view on a group. It isn't a new group, it's just
    a convenient way of referring to a subset of the neurons in an
    already defined group. The subset has to be a continguous set of
    neurons. They can be overlapping if defined with the slice
    notation, or consecutive if defined with the :meth:`subgroup` method.
    Subgroups can themselves be subgrouped. Subgroups can be used in
    almost all situations exactly as if they were groups, except that
    they cannot be passed to the :class:`Network` object.
    
    **Details**
    
    TODO: details of other methods and properties for people
    wanting to write extensions?
    """
#    Group of neurons.
#    Important variables:
#    + S (state variables)
#    + LS (last spike times) (rather use getspikes())
#    + var_index (mapping from variable name -> index in state matrix)
#    Delays can be incorporated
#    + N (number of neurons in group)
    
    @check_units(refractory=second,max_delay=second)
    def __init__(self, N, model=None, threshold=None, reset=NoReset(),
                 init=None, refractory=0*msecond, level=0,
                 clock=None, order=1, implicit=False,unit_checking=True,
                 max_delay=0*msecond, compile=False, freeze=False, method=None,
                 **args):
        '''
        Initializes the group.
        '''

        self._spiking=True # by default, produces spikes
        if bup.use_units: # one additional frame level induced by the decorator
            level+=1

        # Check if model is specified as a Model and load its variables if it is
        if isinstance(model,Model):
            kwds = model.kwds
            for k, v in kwds.items():
                exec k+'=v'

        # If it is a string, convert to Equations object
        if isinstance(model,(str,list,tuple)):
            # level=4 looks pretty strange, but the point is to grab the appropriate namespace from the
            # frame where the string is defined, rather than from the current frame which is what
            # Equations will do by default. The level has to be 4 because it is 1+3, where the 1 is
            # the frame that called the NeuronGroup init, and the 3 is the 3 decorators added to the
            # beginning of the __init__ method.
            model = Equations(model, level=level+1)        

        if isinstance(threshold,str):
            if isinstance(model, Equations):
                threshold = select_threshold(threshold, model, level=level+1)
            else:
                threshold = StringThreshold(threshold, level=level+1)

        if isinstance(reset,str):
            if isinstance(model, Equations):
                reset = select_reset(reset, model, level=level+1)
            else:
                reset = StringReset(reset, level=level+1)

        # Clock
        clock=guess_clock(clock)#not needed with protocol checking
        self.clock=clock 
        
        # Initial state
        self._S0=init
        self.staticvars=[]

        # StateUpdater
        if isinstance(model,StateUpdater):
            self._state_updater=model # Update mechanism
        elif isinstance(model,Equations):
            if (init==None) and (model._units=={}):
                raise AttributeError,"The group must be initialized."
            self._state_updater,var_names=magic_state_updater(model,clock=clock,order=order,check_units=check_units,implicit=implicit,compile=compile,freeze=freeze,method=method)
            Group.__init__(self, model, N)
#            self.staticvars=dict([(name,model._function[name]) for name in model._eq_names])
#            self.var_index=dict(zip(var_names,count()))
#            self.var_index.update(zip(range(len(var_names)),range(len(var_names)))) # name integer i -> state variable i
#            for var1,var2 in model._alias.iteritems():
#                self.var_index[var1]=self.var_index[var2]
            # Converts S0 from dictionary to tuple
            if self._S0==None: # No initialization: 0 with units
                S0={}
            else:
                S0=self._S0.copy()
            # Fill missing units
            for key,value in model._units.iteritems():
                if not(key in S0):
                    S0[key]=0*value
            self._S0=[0]*len(var_names)
            for var,i in zip(var_names,count()):
                self._S0[i]=S0[var]
        else:
            raise TypeError,"StateUpdater must be specified at initialization."        
        # TODO: remove temporary unit hack, this makes all state variables dimensionless if no units are specified
        # What is this??
        if self._S0 is None:
            self._S0 = dict((i,1.) for i in range(len(self._state_updater)))
                 
        # Threshold
        if isinstance(threshold,Threshold):
            self._threshold=threshold
        elif type(threshold)==types.FunctionType:
            if threshold.func_code.co_argcount==1:
                self._threshold=SimpleFunThreshold(threshold)
            else:
                self._threshold=FunThreshold(threshold)
        elif is_scalar_type(threshold):
            # Check unit
            if self._S0!=None:
                try:
                    threshold+self._S0[0]
                except DimensionMismatchError,inst:
                    raise DimensionMismatchError("The threshold does not have correct units.",*inst._dims)
            self._threshold=Threshold(threshold=threshold)
        else: # maybe raise an error?
            self._threshold=NoThreshold()
            self._spiking=False
        
        # Initialization of the state matrix
        if not hasattr(self, '_S'):
            self._S = zeros((len(self._state_updater),N))
        if self._S0!=None:
            for i in range(len(self._state_updater)):
                self._S[i,:]=self._S0[i]
                
        # Reset and refractory period
        if is_scalar_type(reset):
            # Check unit
            if self._S0!=None:
                try:
                    reset+self._S0[0]
                except DimensionMismatchError,inst:
                    raise DimensionMismatchError("The reset does not have correct units.",*inst._dims)
            # What is this 0.9 ?!! Answer: it's just to check that the refractory period is at least clock.dt otherwise don't bother
            if refractory>0.9*clock.dt: # Refractory period - unit checking is done here
                self._resetfun=Refractoriness(period=refractory,resetvalue=reset)
                period=int(refractory/clock.dt)+1
            else: # Simple reset
                self._resetfun=Reset(reset)
                period=1
        elif type(reset)==types.FunctionType:
            self._resetfun=FunReset(reset)
            if refractory>0.9*clock.dt:
                raise ValueError('Refractoriness for custom reset functions not yet implemented, see http://groups.google.fr/group/briansupport/browse_thread/thread/182aaf1af3499a63?hl=en for some options.')
            period=1
        elif hasattr(reset,'period'): # A reset with refractoriness
            # TODO: check unit (in Reset())
            self._resetfun=reset # reset function
            period=int(reset.period/clock.dt)+1
        elif hasattr(threshold,'refractory'): # A threshold with refractoriness
            self._resetfun=reset
            period=threshold.refractory+1 # unit checking done here
        else: # No reset?
            self._resetfun=reset
            period=1
        if max_delay<period*clock.dt:
            max_delay=period*clock.dt
        self._max_delay = 0
        self.period = period
        self.set_max_delay(max_delay)
#        self._max_delay=int(max_delay/clock.dt)+2 # in time bins
#        mp = period-2
#        if mp<1: mp=1
#        self.LS = SpikeContainer(int((N*self._max_delay)/mp)+1,
#                                 self._max_delay,
#                                 useweave=get_global_preference('useweave'),
#                                 compiler=get_global_preference('weavecompiler')) # Spike storage
        
        self._owner=self # owner (for subgroups)
        self._origin=0 # start index from owner if subgroup
        self._next_subgroup=0 # start index of next subgroup
        
        # ensure that var_index has all the 0,...,N-1 integers as names
        if not hasattr(self,'var_index'):
            self.var_index = {}
        for i in range(self.num_states()):
            self.var_index[i] = i
        
        # these are here for the weave accelerated version of the threshold
        # call mechanism.
        self._spikesarray = zeros(N,dtype=int)
        
        # various things for optimising
        self.__t = zeros(N)
        self._var_array = {}
        for i in range(self.num_states()):
            self._var_array[i] = self._S[i]
        for kk, i in self.var_index.iteritems():
            sv = self.state_(i)
            if sv.base is self._S:
                self._var_array[kk] = sv
        
        # state array accessor, only use if var_index array exists
        # todo: should we have a guarantee that var_index exists (even if it just
        # consists of mappings i->i)?
#        if hasattr(self,'var_index'):
#            self.neuron = neuron_array_accessor(self)

    def set_max_delay(self, max_delay):
        _max_delay = int(max_delay/self.clock.dt)+2 # in time bins
        if _max_delay>self._max_delay:
            self._max_delay = _max_delay
            mp = self.period-2
            if mp<1: mp=1
            self.LS = SpikeContainer(int((len(self)*self._max_delay)/mp)+1,
                                     self._max_delay,
                                     useweave=get_global_preference('useweave'),
                                     compiler=get_global_preference('weavecompiler')) # Spike storage

    def rest(self):
        '''
        Sets the variables at rest.
        '''
        self._state_updater.rest(self)

#    def get_var_index(self,name):
#        '''
#        Returns the index of state variable "name".
#        '''
#        return self.var_index[name]

    def reinit(self):
        '''
        Resets the variables.
        '''
        if self._owner==self:
            #self._S[:]=zeros((len(self._state_updater),len(self))) # State matrix
            #self.LS=SpikeContainer(len(self.LS.S),len(self.LS.ind))
            if self._S0!=None:
                for i in range(len(self._state_updater)):
                    self._S[i,:]=self._S0[i]
            else:
                self._S[:]=zeros((len(self._state_updater),len(self))) # State matrix
            self.LS.reinit()
        
    def update(self):
        '''
        Updates the state variables.
        '''
        self._state_updater(self) # update the variables
        if self._spiking:
            spikes=self._threshold(self) # get spikes
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            self.LS.push(spikes) # Store spikes
        
    def get_spikes(self,delay=0):
        '''
        Returns indexes of neurons that spiked at time t-delay*dt.
        '''
        if self._owner==self:
            # Group
#            if delay==0:
#                return self.LS.lastspikes()
            #return self.LS[delay] # last spikes
            return self.LS.get_spikes(delay,0,len(self))
        else:
            # Subgroup
            return self.LS.get_spikes(delay,self._origin,len(self))
#            if delay==0:
#                ls = self.LS.lastspikes()
#            else:
#                ls = self.LS[delay]
            #ls = self.LS[delay]
#            spikes = ls-self._origin
#            return spikes[bisect.bisect_left(spikes,0):\
#                          bisect.bisect_left(spikes,len(self))]
#            return ls[bisect.bisect_left(ls,self._origin):\
#                          bisect.bisect_left(ls,len(self)+self._origin)]-self._origin
    
    def reset(self):
        '''
        Resets the neurons.
        '''
        self._resetfun(self)
        
    def subgroup(self,N):
        if self._next_subgroup+N>len(self):
            raise IndexError,"Subgroup is too large."
        P=self[self._next_subgroup:self._next_subgroup+N]
        self._next_subgroup+=N;
        return P
       
#    def __len__(self):
#        '''
#        Number of neurons in the group.
#        '''
#        return self._S.shape[1]
#
#    def num_states(self):
#        return self._S.shape[0]
    
    def unit(self,name):
        '''
        Returns the unit of variable name
        '''
        if name in self.staticvars:
            f=self.staticvars[name]
            return get_unit(f(*[1.*self.unit(var) for var in f.func_code.co_varnames]))
        elif name=='t': # time
            return second
        else:
            return get_unit(self._S0[self.get_var_index(name)])
    
    def state_(self, name):
        if name=='t':
            self.__t[:] = self.clock._t
            return self.__t
        else:
            return Group.state_(self, name)
    state = state_
    
#    def state_(self,name):
#        '''
#        Gets the state variable named "name" as a reference to the underlying array
#        '''
#        # why doesn't this work?
##        if name in self._var_array:
##            return self._var_array[name]
#        if isinstance(name,int):
#            return self._S[name]
#        if name=='t':
#            self.__t[:] = self.clock._t
#            return self.__t
#        if name in self.staticvars:
#            f=self.staticvars[name]
#            return f(*[self.state_(var) for var in f.func_code.co_varnames])
#        i=self.var_index[name]
#        return self._S[i]
#    state = state_

#    def state(self,name):
#        '''
#        Gets the state variable named "name" as a safe qarray
#        [Romain: I got rid of safeqarray here, which makes a huge speed difference!]
#        '''
#        return self.state_(name)
#        #if name=='t':
#        #    return safeqarray(self.state_('t'),units=second)
#        #return safeqarray(self.state_(name),units=self.unit(name))
    
    def __getitem__(self,i):
        return self[i:i+1]
    
#    def __getattr__(self,name):
#        if name=='var_index':
#            # this seems mad - the reason is that getattr is only called if the thing hasn't
#            # been found using the standard methods of finding attributes, which for var_index
#            # should have worked, this is important because the next line looks for var_index
#            # and if we haven't got a var_index we don't want to get stuck in an infinite
#            # loop
#            raise AttributeError  
#        if not hasattr(self,'var_index'):
#            # only provide lookup of variable names if we have some variable names, i.e.
#            # if the var_index attribute exists
#            raise AttributeError
#        try:
#            return self.state(name)
#        except KeyError:
#            if len(name) and name[-1]=='_':
#                try:
#                    origname = name[:-1]
#                    return self.state_(origname)
#                except KeyError:
#                    raise AttributeError
#            raise AttributeError
#    
#    def __setattr__(self,name,val):
#        origname = name
#        if len(name) and name[-1]=='_':
#            origname = name[:-1]
#        if not hasattr(self,'var_index') or (name not in self.var_index and origname not in self.var_index):
#            object.__setattr__(self,name,val)
#            if not hasattr(self,'_setattrcount'):
#                object.__setattr__(self,'_setattrcount',0)
#            object.__setattr__(self,'_setattrcount',self._setattrcount+1)
#        else:
#            if name in self.var_index:
#                self.state(name)[:]=val
#            else:
#                self.state_(origname)[:]=val

    def __getslice__(self,i,j):
        '''
        Creates subgroup (view).
        TODO: views for all arrays.
        '''
        Q=copy.copy(self)
        Q._S=self._S[:,i:j]
        Q.N=Q._S.shape[1]
        Q._origin=self._origin+i
        Q._next_subgroup = 0
        return Q
    
    def same(self,Q):
        '''
        Tests if the two groups (subgroups) are of the same kind,
        i.e., if they can be added.
        This is not used at the moment.
        '''
        # Same class?
        if self.__class__!=Q.__class__:
            return False
        # Check all variables except arrays and a few ones
        exceptvar=['owner','nextsubgroup','origin']
        for v,val in self.__dict__.iteritems():
            if not(v in Q.__dict__):
                return False
            if (not(isinstance(val,ndarray)) and (not v in exceptvar) and (val!=Q.__dict__[v])):
                return False
        for v in Q.__dict__.iterkeys():
            if not(v in self.__dict__):
                return False
            
        return True
    
    def __repr__(self):
        if self._owner==self:
            return 'Group of '+str(len(self))+' neurons'
        else:
            return 'Subgroup of '+str(len(self))+' neurons'


class PoissonGroup(NeuronGroup):
    '''
    A group that generates independent Poisson spike trains.
    
    **Initialised as:** ::
    
        PoissonGroup(N,rates[,clock])
    
    with arguments:
    
    ``N``
        The number of neurons in the group
    ``rates``
        A scalar, array or function returning a scalar or array.
        The array should have the same length as the number of
        neurons in the group. The function should take one
        argument ``t`` the current simulation time.
    ``clock``
        The clock which the group will update with, do not
        specify to use the default clock.
    '''
    def __init__(self,N,rates=0*hertz,clock=None):
        '''
        Initializes the group.
        P.rates gives the rates.
        '''
        NeuronGroup.__init__(self,N,model=LazyStateUpdater(),threshold=PoissonThreshold(),\
                             clock=clock)
        if callable(rates): # a function is passed
            self._variable_rate=True
            self.rates=rates
            self._S0[0]=self.rates(self.clock.t)
        else:
            self._variable_rate=False
            self._S[0,:]=rates
            self._S0[0]=rates
        self.var_index={'rate':0}

    def update(self):
        if self._variable_rate:
            self._S[0,:]=self.rates(self.clock.t)
        NeuronGroup.update(self)