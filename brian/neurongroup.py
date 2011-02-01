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
__all__ = ['NeuronGroup', 'linked_var']

from numpy import *
from scipy import rand, linalg, random
from numpy.random import exponential, randint
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
import sys
from brian_unit_prefs import bup
import numpy
from base import *
from group import *
from threshold import select_threshold
from collections import defaultdict

timedarray = None # ugly hack: import this module when it is needed, can't do it here because of order of imports
network = None # ugly hack: import this module when it is needed, can't do it here because of order of imports


class TArray(numpy.ndarray):
    '''
    This internal class is just used for when Brian sends an array t
    to an object. All the elements will be the same in this case, and
    you can check for isinstance(arr, TArray) to do optimisations based
    on this. This behaviour may change in the future.
    '''
    def __new__(subtype, arr):
        # All numpy.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        return numpy.array(arr, copy=False).view(subtype)


class LinkedVar(object):
    def __init__(self, source, var=0, func=None, when='start', clock=None):
        self.source = source
        self.var = var
        self.func = func
        self.when = when
        self.clock = clock

def linked_var(source, var=0, func=None, when='start', clock=None):
    """
    Used for linking one :class:`NeuronGroup` variable to another.
    
    Sample usage::
    
        G = NeuronGroup(...)
        H = NeuronGroup(...)
        G.V = linked_var(H, 'W')

    In this scenario, the variable V in group G will always be updated with
    the values from variable W in group H. The groups G and H must be the
    same size (although subgroups can be used if they are not the same size).
    
    Arguments:
    
    ``source``
        The group from which values will be taken.
    ``var``
        The state variable of the source group to take values from.
    ``func``
        An additional function of one argument to pass the source variable
        values through, e.g. ``func=lambda x:clip(x,0,Inf)`` to half rectify the
        values.
    ``when``
        The time in the main Brian loop at which the copy operation is performed,
        as explained in :class:`Network`.
    ``clock``
        The update clock for the copy operation, by default it will use the clock
        of the target group.
    """
    return LinkedVar(source, var, func, when, clock)


class NeuronGroup(magic.InstanceTracker, ObjectContainer, Group):
    """Group of neurons
    
    Initialised with arguments:
    
    ``N``
        The number of neurons in the group.
    ``model``
        An object defining the neuron model. It can be
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
    ``refractory=0*ms``, ``min_refractory``, ``max_refractory``
        A refractory period, used in combination with the ``reset`` value
        if it is a scalar. For constant resets only, you can specify refractory
        as an array of length the number of elements in the group, or as a
        string, giving the name of a state variable in the group. In the case
        of these variable refractory periods, you should specify
        ``min_refractory`` (optional) and ``max_refractory`` (required).
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
        above. Writing ``G.x = arr`` for ``arr`` a :class:`TimedArray`
        will set the values of variable x to be ``arr(t)`` at time t.
        See :class:`TimedArraySetter` for details. 
    
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

    @check_units(max_delay=second)
    def __init__(self, N, model=None, threshold=None, reset=NoReset(),
                 init=None, refractory=0 * msecond, level=0,
                 clock=None, order=1, implicit=False, unit_checking=True,
                 max_delay=0 * msecond, compile=False, freeze=False, method=None,
                 max_refractory=None,
                 ):#**args): # any reason why **args was included here?
        '''
        Initializes the group.
        '''

        self._spiking = True # by default, produces spikes
        if bup.use_units: # one additional frame level induced by the decorator
            level += 1

        # If it is a string, convert to Equations object
        if isinstance(model, (str, list, tuple)):
            model = Equations(model, level=level + 1)

        if isinstance(threshold, str):
            if isinstance(model, Equations):
                threshold = select_threshold(threshold, model, level=level + 1)
            else:
                threshold = StringThreshold(threshold, level=level + 1)

        if isinstance(reset, str):
            if isinstance(model, Equations):
                reset = select_reset(reset, model, level=level + 1)
            else:
                reset = StringReset(reset, level=level + 1)

        # Clock
        clock = guess_clock(clock)#not needed with protocol checking
        self.clock = clock

        # Initial state
        self._S0 = init
        self.staticvars = []

        # StateUpdater
        if isinstance(model, StateUpdater):
            self._state_updater = model # Update mechanism
            self._all_units = defaultdict()
        elif isinstance(model, Equations):
            self._eqs = model
            if (init == None) and (model._units == {}):
                raise AttributeError, "The group must be initialized."
            self._state_updater, var_names = magic_state_updater(model, clock=clock, order=order,
                                                                 check_units=unit_checking, implicit=implicit,
                                                                 compile=compile, freeze=freeze,
                                                                 method=method)
            Group.__init__(self, model, N, unit_checking=unit_checking)
            self._all_units = model._units
            # Converts S0 from dictionary to tuple
            if self._S0 == None: # No initialization: 0 with units
                S0 = {}
            else:
                S0 = self._S0.copy()
            # Fill missing units
            for key, value in model._units.iteritems():
                if not key in S0:
                    S0[key] = 0 * value
            self._S0 = [0] * len(var_names)
            for var, i in zip(var_names, count()):
                self._S0[i] = S0[var]
        else:
            raise TypeError, "StateUpdater must be specified at initialization."
        # TODO: remove temporary unit hack, this makes all state variables dimensionless if no units are specified
        # What is this??
        if self._S0 is None:
            self._S0 = dict((i, 1.) for i in range(len(self._state_updater)))

        # Threshold
        if isinstance(threshold, Threshold):
            self._threshold = threshold
        elif type(threshold) == types.FunctionType:
            if threshold.func_code.co_argcount == 1:
                self._threshold = SimpleFunThreshold(threshold)
            else:
                self._threshold = FunThreshold(threshold)
        elif is_scalar_type(threshold):
            # Check unit
            if self._S0 != None:
                try:
                    threshold + self._S0[0]
                except DimensionMismatchError, inst:
                    raise DimensionMismatchError("The threshold does not have correct units.", *inst._dims)
            self._threshold = Threshold(threshold=threshold)
        else: # maybe raise an error?
            self._threshold = NoThreshold()
            self._spiking = False

        # Initialization of the state matrix
        if not hasattr(self, '_S'):
            self._S = zeros((len(self._state_updater), N))
        if self._S0 != None:
            for i in range(len(self._state_updater)):
                self._S[i, :] = self._S0[i]

        # Reset and refractory period
        self._variable_refractory_time = False
        period_max = 0
        if is_scalar_type(reset) or reset.__class__ is Reset:
            if reset.__class__ is Reset:
                if isinstance(reset.state, str):
                    numstate = self.get_var_index(reset.state)
                else:
                    numstate = reset.state
                reset = reset.resetvalue
            else:
                numstate = 0
            # Check unit
            if self._S0 != None:
                try:
                    reset + self._S0[numstate]
                except DimensionMismatchError, inst:
                    raise DimensionMismatchError("The reset does not have correct units.", *inst._dims)
            if isinstance(refractory, float):
                max_refractory = refractory
            else:
                if isinstance(refractory, str):
                    if max_refractory is None:
                        raise ValueError('Must specify max_refractory if using variable refractoriness.')
                    self._refractory_variable = refractory
                    self._refractory_array = None
                else:
                    max_refractory = amax(refractory) * second
                    self._refractory_variable = None
                    self._refractory_array = refractory
                self._variable_refractory_time = True
            # What is this 0.9 ?!! Answer: it's just to check that the refractory period is at least clock.dt otherwise don't bother
            if max_refractory > 0.9 * clock.dt: # Refractory period - unit checking is done here
                self._resetfun = Refractoriness(period=max_refractory, resetvalue=reset, state=numstate)
                period_max = int(max_refractory / clock.dt) + 1
            else: # Simple reset
                self._resetfun = Reset(reset, state=numstate)
        elif type(reset) == types.FunctionType:
            self._resetfun = FunReset(reset)
            if refractory > 0.9 * clock.dt:
                raise ValueError('Refractoriness for custom reset functions not yet implemented, see http://groups.google.fr/group/briansupport/browse_thread/thread/182aaf1af3499a63?hl=en for some options.')
        elif hasattr(reset, 'period'): # A reset with refractoriness
            # TODO: check unit (in Reset())
            self._resetfun = reset # reset function
            period_max = int(reset.period / clock.dt) + 1
        else: # No reset?
            self._resetfun = reset
        if hasattr(threshold, 'refractory'): # A threshold with refractoriness
            period_max = max(period_max, threshold.refractory + 1)
        if max_refractory is None:
            max_refractory = refractory
        if max_delay < period_max * clock.dt:
            max_delay = period_max * clock.dt
        self._max_delay = 0
        self.set_max_delay(max_delay)

        self._next_allowed_spiketime = -ones(N)
        self._refractory_time = float(max_refractory) - 0.5 * clock._dt
        self._use_next_allowed_spiketime_refractoriness = True

        self._owner = self # owner (for subgroups)
        self._subgroup_set = magic.WeakSet()
        self._origin = 0 # start index from owner if subgroup
        self._next_subgroup = 0 # start index of next subgroup

        # ensure that var_index has all the 0,...,N-1 integers as names
        if not hasattr(self, 'var_index'):
            self.var_index = {}
        for i in range(self.num_states()):
            self.var_index[i] = i

        # these are here for the weave accelerated version of the threshold
        # call mechanism.
        self._spikesarray = zeros(N, dtype=int)

        # various things for optimising
        self.__t = TArray(zeros(N))
        self._var_array = {}
        for i in range(self.num_states()):
            self._var_array[i] = self._S[i]
        for kk, i in self.var_index.iteritems():
            sv = self.state_(i)
            if sv.base is self._S:
                self._var_array[kk] = sv

        # todo: should we have a guarantee that var_index exists (even if it just
        # consists of mappings i->i)?

    def set_max_delay(self, max_delay):
        if hasattr(self, '_owner') and self._owner is not self:
            self._owner.set_max_delay(max_delay)
            return
        _max_delay = int(max_delay / self.clock.dt) + 2 # in time bins
        if _max_delay > self._max_delay:
            self._max_delay = _max_delay
            self.LS = SpikeContainer(self._max_delay,
                                     useweave=get_global_preference('useweave'),
                                     compiler=get_global_preference('weavecompiler')) # Spike storage
            # update all subgroups if any exist
            if hasattr(self, '_subgroup_set'): # the first time set_max_delay is called this is false
                for G in self._owner._subgroup_set.get():
                    G._max_delay = self._max_delay
                    G.LS = self.LS

    def rest(self):
        '''
        Sets the variables at rest.
        '''
        self._state_updater.rest(self)

    def reinit(self, states=True):
        '''
        Resets the variables.
        '''
        if self._owner is self:
            if states:
                if self._S0 is not None:
                    for i in range(len(self._state_updater)):
                        self._S[i, :] = self._S0[i]
                else:
                    self._S[:] = 0 # State matrix
            self._next_allowed_spiketime[:] = -1
            self.LS.reinit()

    def update(self):
        '''
        Updates the state variables.
        '''
        self._state_updater(self) # update the variables
        if self._spiking:
            spikes = self._threshold(self) # get spikes
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            if self._use_next_allowed_spiketime_refractoriness:
                spikes = spikes[self._next_allowed_spiketime[spikes] <= self.clock._t]
                if self._variable_refractory_time:
                    if self._refractory_variable is not None:
                        refractime = self.state_(self._refractory_variable)
                    else:
                        refractime = self._refractory_array
                    self._next_allowed_spiketime[spikes] = self.clock._t + refractime[spikes]
                else:
                    self._next_allowed_spiketime[spikes] = self.clock._t + self._refractory_time
            self.LS.push(spikes) # Store spikes

    def get_refractory_indices(self):
        return (self._next_allowed_spiketime > self.clock._t).nonzero()[0]

    def get_spikes(self, delay=0):
        '''
        Returns indexes of neurons that spiked at time t-delay*dt.
        '''
        if self._owner == self:
            # Group
#            if delay==0:
#                return self.LS.lastspikes()
            #return self.LS[delay] # last spikes
            return self.LS.get_spikes(delay, 0, len(self))
        else:
            # Subgroup
            return self.LS.get_spikes(delay, self._origin, len(self))
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

    def subgroup(self, N):
        if self._next_subgroup + N > len(self):
            raise IndexError, "Subgroup is too large."
        P = self[self._next_subgroup:self._next_subgroup + N]
        self._next_subgroup += N;
        return P

    def unit(self, name):
        '''
        Returns the unit of variable name
        '''
        if name in self._all_units:
            return self._all_units[name]
        elif name in self.staticvars:
            f = self.staticvars[name]
            print f.func_code.co_varnames
            print [(var, self.unit(var)) for var in f.func_code.co_varnames]
            return get_unit(f(*[1. * self.unit(var) for var in f.func_code.co_varnames]))
        elif name == 't': # time
            return second
        else:
            return get_unit(self._S0[self.get_var_index(name)])

    def state_(self, name):
        if name == 't':
            self.__t[:] = self.clock._t
            return self.__t
        else:
            return Group.state_(self, name)
    state = state_

    def __getitem__(self, i):
        if i == -1:
            return self[self._S.shape[1] - 1:]
        else:
            return self[i:i + 1]

    def __getslice__(self, i, j):
        '''
        Creates subgroup (view).
        TODO: views for all arrays.
        '''
        Q = copy.copy(self)
        Q._S = self._S[:, i:j]
        Q.N = Q._S.shape[1]
        Q._origin = self._origin + i
        Q._next_subgroup = 0
        self._subgroup_set.add(Q)
        return Q

    def same(self, Q):
        '''
        Tests if the two groups (subgroups) are of the same kind,
        i.e., if they can be added.
        This is not used at the moment.
        OBSOLETE
        '''
        # Same class?
        if self.__class__ != Q.__class__:
            return False
        # Check all variables except arrays and a few ones
        exceptvar = ['owner', 'nextsubgroup', 'origin']
        for v, val in self.__dict__.iteritems():
            if not(v in Q.__dict__):
                return False
            if (not(isinstance(val, ndarray)) and (not v in exceptvar) and (val != Q.__dict__[v])):
                return False
        for v in Q.__dict__.iterkeys():
            if not(v in self.__dict__):
                return False

        return True

    def __repr__(self):
        if self._owner == self:
            return 'Group of ' + str(len(self)) + ' neurons'
        else:
            return 'Subgroup of ' + str(len(self)) + ' neurons'

    def __setattr__(self, name, val):
        global timedarray
        if timedarray is None:
            import timedarray
        if isinstance(val, timedarray.TimedArray):
            self.set_var_by_array(name, val)
        elif isinstance(val, LinkedVar):
            self.link_var(name, val.source, val.var, val.func, val.when, val.clock)
        else:
            Group.__setattr__(self, name, val)

    def link_var(self, var, source, sourcevar, func=None, when='start', clock=None):
        global network
        if network is None:
            import network
        if clock is None:
            clock = self.clock
        # check that var is not an equation (it really should only be a parameter
        # but not sure how to make this generic and still work with neurongroups
        # that aren't defined by Equations objects)
        if hasattr(self, 'staticvars') and var in self.staticvars:
            raise ValueError("Cannot set a static variable (equation) with a linked variable.")
        selfarr = self.state_(var)
        if hasattr(source, 'staticvars') and sourcevar in source.staticvars:
            if func is None: func = lambda x: x
            @network.network_operation(when=when, clock=clock)
            def update_link_var():
                selfarr[:] = func(getattr(source, sourcevar))
        else:
            sourcearr = source.state_(sourcevar)
            if func is None:
                @network.network_operation(when=when, clock=clock)
                def update_link_var():
                    selfarr[:] = sourcearr
            else:
                @network.network_operation(when=when, clock=clock)
                def update_link_var():
                    selfarr[:] = func(sourcearr)
        self._owner.contained_objects.append(update_link_var)

    def set_var_by_array(self, var, arr, times=None, clock=None, start=None, dt=None):
        # ugly hack, have to import this here because otherwise the order of imports
        # is messed up.
        import timedarray
        timedarray.set_group_var_by_array(self, var, arr, times=times, clock=clock, start=start, dt=dt)
