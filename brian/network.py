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
Network class
'''
__all__ = ['Network', 'MagicNetwork', 'NetworkOperation', 'network_operation', 'run',
           'reinit', 'stop', 'clear', 'forget', 'recall']

from Queue import Queue
from connections import *
from neurongroup import NeuronGroup
from clock import guess_clock, Clock
import magic
from inspect import *
from operator import isSequenceType
import types
from itertools import chain
from collections import defaultdict
import copy
from base import *
from units import second
import time
from utils.progressreporting import *
from globalprefs import *
import gc
import heapq

globally_stopped = False


class Network(object):
    '''
    Contains simulation objects and runs simulations
    
    **Initialised as:** ::
    
        Network(...)
    
    with ``...`` any collection of objects that should be added to the :class:`Network`.
    You can also pass lists of objects, lists of lists of objects, etc. Objects
    that need to passed to the :class:`Network` object are:
    
    * :class:`NeuronGroup` and anything derived from it such as :class:`PoissonGroup`.
    * :class:`Connection` and anything derived from it.
    * Any monitor such as :class:`SpikeMonitor` or :class:`StateMonitor`.
    * Any network operation defined with the :func:`network_operation` decorator.
    
    Models, equations, etc. do not need to be passed to the :class:`Network` object. 
    
    The most important method is the ``run(duration)`` method which runs the simulation
    for the given length of time (see below for details about what happens when you
    do this).
    
    **Example usage:** ::
    
        G = NeuronGroup(...)
        C = Connection(...)
        net = Network(G,C)
        net.run(1*second)
    
    **Methods**
    
    ``add(...)``
        Add additional objects after initialisation, works the same way
        as initialisation.
    ``run(duration[, report[, report_period]])``
        Runs the network for the given duration. See below for details about
        what happens when you do this. See documentation for :func:`run` for
        an explanation of the ``report`` and ``report_period`` keywords.
    ``reinit(states=True)``
        Reinitialises the network, runs each object's ``reinit()`` and each
        clock's ``reinit()`` method (resetting them to 0). If ``states=False``
        then it will not reinitialise the :class:`NeuronGroup` state variables.
    ``stop()``
        Can be called from a :func:`network_operation` for example to stop the
        network from running.
    ``__len__()``
        Returns the number of neurons in the network.
    ``__call__(obj)``
        Similar to ``add``, but you can only pass one object and that
        object is returned. You would only need this in obscure
        circumstances where objects needed to be added to the network
        but were either not stored elsewhere or were stored in a way
        that made them difficult to extract, for example below the
        NeuronGroup object is only added to the network if certain
        conditions hold::
        
            net = Network(...)
            if some_condition:
                x = net(NeuronGroup(...))
    
    **What happens when you run**
    
    For an overview, see the Concepts chapter of the main documentation.
    
    When you run the network, the first thing that happens is that it
    checks if it has been prepared and calls the ``prepare()`` method
    if not. This just does various housekeeping tasks and optimisations
    to make the simulation run faster. Also, an update schedule is
    built at this point (see below).
    
    Now the ``update()`` method is repeatedly called until every clock
    has run for the given length of time. After each call of the
    ``update()`` method, the clock is advanced by one tick, and if
    multiple clocks are being used, the next clock is determined (this
    is the clock whose value of ``t`` is minimal amongst all the clocks).
    For example, if you had two clocks in operation, say ``clock1`` with
    ``dt=3*ms`` and ``clock2`` with ``dt=5*ms`` then this will happen:
    
    1. ``update()`` for ``clock1``, tick ``clock1`` to ``t=3*ms``, next
       clock is ``clock2`` with ``t=0*ms``.
    2. ``update()`` for ``clock2``, tick ``clock2`` to ``t=5*ms``, next
       clock is ``clock1`` with ``t=3*ms``.
    3. ``update()`` for ``clock1``, tick ``clock1`` to ``t=6*ms``, next
       clock is ``clock2`` with ``t=5*ms``.
    4. ``update()`` for ``clock2``, tick ``clock2`` to ``t=10*ms``, next
       clock is ``clock1`` with ``t=6*ms``.
    5. ``update()`` for ``clock1``, tick ``clock1`` to ``t=9*ms``, next
       clock is ``clock1`` with ``t=9*ms``.
    6. ``update()`` for ``clock1``, tick ``clock1`` to ``t=12*ms``, next
       clock is ``clock2`` with ``t=10*ms``. etc.
    
    The ``update()`` method simply runs each operation in the current clock's
    update schedule. See below for details on the update schedule.
    
    **Update schedules**
    
    An update schedule is the sequence of operations that are
    called for each ``update()`` step. The standard update schedule is:
    
    *  Network operations with ``when = 'start'``
    *  Network operations with ``when = 'before_groups'``
    *  Call ``update()`` method for each :class:`NeuronGroup`, this typically
       performs an integration time step for the differential equations
       defining the neuron model.
    *  Network operations with ``when = 'after_groups'``
    *  Network operations with ``when = 'middle'``
    *  Network operations with ``when = 'before_connections'``
    *  Call ``do_propagate()`` method for each :class:`Connection`, this
       typically adds a value to the target state variable of each neuron
       that a neuron that has fired is connected to. See Tutorial 2: Connections for
       a more detailed explanation of this.
    *  Network operations with ``when = 'after_connections'``
    *  Network operations with ``when = 'before_resets'``
    *  Call ``reset()`` method for each :class:`NeuronGroup`, typically resets a
       given state variable to a given reset value for each neuron that fired
       in this update step.
    *  Network operations with ``when = 'after_resets'``
    *  Network operations with ``when = 'end'``
    
    There is one predefined alternative schedule, which you can choose by calling
    the ``update_schedule_groups_resets_connections()`` method before running the
    network for the first time. As the name suggests, the reset operations are
    done before connections (and the appropriately named network operations are
    called relative to this rearrangement). You can also define your own update
    schedule with the ``set_update_schedule`` method (see that method's API documentation for
    details). This might be useful for example if you have a sequence of network
    operations which need to be run in a given order.    
    '''

    operations = property(fget=lambda self:self._all_operations)

    def __init__(self, *args, **kwds):
        self.clock = None # Initialized later
        self.groups = []
        self.connections = []
        # The following dict keeps a copy of which operations are in which slot
        self._operations_dict = defaultdict(list)
        self._all_operations = []
        self.update_schedule_standard()
        self.prepared = False
        for o in chain(args, kwds.itervalues()):
            self.add(o)

    def add(self, *objs):
        """
        Add an object or container of objects to the network
        """
        for obj in objs:
            if isinstance(obj, NeuronGroup):
                if obj not in self.groups:
                    self.groups.append(obj)
            elif isinstance(obj, Connection):
                if obj not in self.connections:
                    self.connections.append(obj)
            elif isinstance(obj, NetworkOperation):
                if obj not in self._all_operations:
                    self._operations_dict[obj.when].append(obj)
                    self._all_operations.append(obj)
            elif isSequenceType(obj):
                for o in obj:
                    self.add(o)
            else:
                raise TypeError('Only the following types of objects can be added to a network: NeuronGroup, Connection or NetworkOperation')

            try:
                gco = obj.contained_objects
                if gco is not None:
                    self.add(gco)
            except AttributeError:
                pass

    def __call__(self, obj):
        """
        Add an object to the network and return it
        """
        self.add(obj)
        return obj

    def reinit(self, states=True):
        '''
        Resets the objects and clocks. If ``states=False`` it will not reinit
        the state variables.
        '''
        objs = self.groups + self.connections + self.operations
        if self.clock is not None:
            objs.append(self.clock)
        else:
            guess_clock(None).reinit()
        if hasattr(self, 'clocks'):
            objs.extend(self.clocks)
        for P in objs:
            if hasattr(P, 'reinit'):
                if isinstance(P, NeuronGroup):
                    try:
                        P.reinit(states=states)
                    except TypeError:
                        P.reinit()
                else:
                    P.reinit()

    def prepare(self):
        '''
        Prepares the network for simulation:
        + Checks the clocks of the neuron groups
        + Gather connections with identical subgroups
        + Compresses the connection matrices for faster simulation
        Calling this function is not mandatory but speeds up the simulation.
        '''
        # Set the clock
        if self.same_clocks():
            self.set_clock()
        else:
            self.set_many_clocks()

        # Gather connections with identical subgroups
        # 'subgroups' maps subgroups to connections (initialize with immutable object (not [])!)
        subgroups = dict.fromkeys([(C.source, C.delay) for C in self.connections], None)
        for C in self.connections:
            if subgroups[(C.source, C.delay)] == None:
                subgroups[(C.source, C.delay)] = [C]
            else:
                subgroups[(C.source, C.delay)].append(C)
        self.connections = subgroups.values()
        cons = self.connections # just for readability
        for i in range(len(cons)):
            if len(cons[i]) > 1: # at least 2 connections with the same subgroup
                cons[i] = MultiConnection(cons[i][0].source, cons[i])
            else:
                cons[i] = cons[i][0]

        # Compress connections
        for C in self.connections:
            C.compress()

        # Experimental support for new propagation code
        if get_global_preference('usenewpropagate') and get_global_preference('useweave'):
            from experimental.new_c_propagate import make_new_connection
            for C in self.connections:
                make_new_connection(C)

        # build operations list for each clock
        self._build_update_schedule()

        self.prepared = True

    def update_schedule_standard(self):
        self._schedule = ['ops start',
                          'ops before_groups',
                          'groups',
                          'ops after_groups',
                          'ops middle',
                          'ops before_connections',
                          'connections',
                          'ops after_connections',
                          'ops before_resets',
                          'resets',
                          'ops after_resets',
                          'ops end'
                          ]
        self._build_update_schedule()

    def update_schedule_groups_resets_connections(self):
        self._schedule = ['ops start',
                          'ops before_groups',
                          'groups',
                          'ops after_groups',
                          'ops middle',
                          'ops before_resets',
                          'resets',
                          'ops after_resets',
                          'ops before_connections',
                          'connections',
                          'ops after_connections',
                          'ops end'
                          ]
        self._build_update_schedule()

    def set_update_schedule(self, schedule):
        """
        Defines a custom update schedule
        
        A custom update schedule is a list of schedule items. Each update
        step of the network, the schedule items will be run in turn. A
        schedule item can be defined as a string or tuple. The following
        string definitions are possible:
        
        'groups'
            Calls the 'update' function of each group in turn, this is
            typically the integration step of the simulation.
        'connections'
            Calls the 'do_propagate' function of each connection in
            turn, this is typically propagating spikes forward (and
            backward in the case of STDP).
        'resets'
            Calls the 'reset' function of each group in turn.
        'ops '+name
            Calls each operation in turn whose 'when' parameter is
            set to 'name'. The standard set of 'when' names is
            start, before_groups, after_groups, before_resets,
            after_resets, before_connections, after_connections,
            end, but you can use any you like.
        
        If a tuple is provided, it should be of the form:
        
            (objset, func, allclocks)
        
        with:
        
        objset
            a list of objects to be processed
        func
            Either None or a string. In the case of none, each
            object in objset must be callable and will be called.
            In the case of a string, obj.func will be called for
            each obj in objset.
        allclocks
            Either True or False. If it's set to True, then the
            object will be placed in the update schedule of
            every clock in the network. If False, it will be
            placed in the update schedule only of the clock
            obj.clock.
        """
        self._schedule = schedule
        self._build_update_schedule()

    def _build_update_schedule(self):
        '''
        Defines what the update step does
        
        For each clock we build a list self._update_schedule[id(clock)]
        of functions which are called at the update step if that clock
        is active. This is generic and works for single or multiple
        clocks.
        
        See documentation for set_update_schedule for an explanation of
        the self._schedule object. 
        '''
        self._update_schedule = defaultdict(list)
        if hasattr(self, 'clocks'):
            clocks = self.clocks
        else:
            clocks = [self.clock]
        clockset = clocks
        for item in self._schedule:
            # we define some simple names for common schedule items
            if isinstance(item, str):
                if item == 'groups':
                    objset = self.groups
                    objfun = 'update'
                    allclocks = False
                elif item == 'resets':
                    objset = self.groups
                    objfun = 'reset'
                    allclocks = False
                elif item == 'connections':
                    objset = self.connections
                    objfun = 'do_propagate'
                    allclocks = False
                    # Connections do not define their own clock, but they should
                    # be updated on the schedule of their source group
                    for obj in objset:
                        obj.clock = obj.source.clock
                elif len(item) > 4 and item[0:3] == 'ops': # the item is of the forms 'ops when'
                    objset = self._operations_dict[item[4:]]
                    objfun = None
                    allclocks = False
            else:
                # we allow the more general form of usage as well
                objset, objfun, allclocks = item
            for obj in objset:
                if objfun is None:
                    f = obj
                else:
                    f = getattr(obj, objfun)
                if not allclocks:
                    useclockset = [obj.clock]
                else:
                    useclockset = clockset
                for clock in useclockset:
                    self._update_schedule[id(clock)].append(f)

    def update(self):
        for f in self._update_schedule[id(self.clock)]:
            f()

    """
    def update_threaded(self,queue):
        '''
        EXPERIMENTAL (not useful for the moment)
        Parallel update of the network (using threads).
        '''
        # Update groups: one group = one thread
        for P in self.groups:
            queue.put(P)
        queue.join() # Wait until job is done

        # The following is done serially
        # Propagate spikes
        for C in self.connections:
            C.propagate(C.source.get_spikes(C.delay))
            
        # Miscellanous operations
        for op in self.operations:
            op()
    """

    def run(self, duration, threads=1, report=None, report_period=10 * second):
        '''
        Runs the simulation for the given duration.
        '''
        global globally_stopped
        self.stopped = False
        globally_stopped = False
        if not self.prepared:
            self.prepare()
        self.clock.set_duration(duration)
        try:
            for c in self.clocks:
                c.set_duration(duration)
        except AttributeError:
            pass
        if report is not None:
            start_time = time.time()
            if not isinstance(report, ProgressReporter):
                report = ProgressReporter(report, report_period)
                next_report_time = start_time + float(report_period)
            else:
                report_period = report.period
                next_report_time = report.next_report_time

        if self.clock.still_running() and not self.stopped and not globally_stopped:
            not_same_clocks = not self.same_clocks()
            clk = self.clock
            while clk.still_running() and not self.stopped and not globally_stopped:
                if report is not None:
                    cur_time = time.time()
                    if cur_time > next_report_time:
                        next_report_time = cur_time + float(report_period)
                        report.update((self.clock.t - self.clock.start) / duration)
                self.update()
                clk.tick()
                if not_same_clocks:
                    # Find the next clock to update
                    #self.clock = min([(clock.t, id(clock), clock) for clock in self.clocks])[2]
                    clk = self.clock = min(self.clocks)
                    #heapq.heappush(self.clocks, self.clock)
                    #clk = self.clock = heapq.heappop(self.clocks)
                    #clk = self.clock = heapq.heappushpop(self.clocks, self.clock)
                    #if not clk<self.clocks[0]:
                    #    clk = self.clock = heapq.heapreplace(self.clocks, self.clock)
        if report is not None:
            report.update(1.0)

    def stop(self):
        '''
        Stops the network from running, this is reset the next time ``run()`` is called.
        '''
        self.stopped = True

    def same_clocks(self):
        '''
        Returns True if the clocks of all groups and operations are the same.
        '''
        groups_and_operations=self.groups + self.operations
        if len(groups_and_operations)>0:
            clock = groups_and_operations[0].clock
            return all([obj.clock == clock for obj in groups_and_operations])
        else:
            return True

    def set_clock(self):
        '''
        Sets the clock and checks that clocks of all groups are synchronized.
        '''
        if self.same_clocks():
            groups_and_operations=self.groups + self.operations
            if len(groups_and_operations)>0:
                self.clock = groups_and_operations[0].clock
            else:
                self.clock = guess_clock()
        else:
            raise TypeError, 'Clocks are not synchronized!' # other error type?

    def set_many_clocks(self):
        '''
        Sets a list of clocks.
        self.clock points to the current clock between considered.
        '''
        self.clocks = list(set([obj.clock for obj in self.groups + self.operations]))
        #self.clocks.sort(key=lambda c:-c._dt)
        #heapq.heapify(self.clocks)
        #self.clock = min([(clock.t, clock) for clock in self.clocks])[1]
        self.clock = min(self.clocks)
        #self.clock = heapq.heappop(self.clocks)

    def __len__(self):
        '''
        Number of neurons in the network
        '''
        n = 0
        for P in self.groups:
            n += len(P) # use compact iterator function?
        return n

    def __repr__(self):
        return 'Network of' + str(len(self)) + 'neurons'

    # TODO: obscure custom update schedules might still lead to unpicklable Network object
    def __reduce__(self):
        # This code might need some explanation:
        #
        # The problem with pickling the Network object is that you cannot pickle
        # 'instance methods', that is a copy of a method of an instance. The
        # Network object does this because the _update_schedule attribute stores
        # a copy of all the functions that need to be called each time step, and
        # these are all instance methods (of NeuronGroup, Reset, etc.). So, we
        # solve the problem by deleting this attribute at pickling time, and then
        # rebuilding it at unpickling time. The function unpickle_network defined
        # below does the unpickling.
        #
        # We basically want to make a copy of the current object, and delete the
        # update schedule from it, and then pickle that. Some weird recursive
        # stuff happens if you try to do this in the obvious way, so we take the
        # seemingly mad step of converting the object to a general 'heap' class
        # (that is, a new-style class with no methods or anything, in this case
        # the NetworkNoMethods class defined below), do all our operations on
        # this, store a copy of the actual class of the object (which may not be
        # Network for derived classes), work with this, and then restore
        # everything back to the way it was when everything is done.
        #
        oldclass = self.__class__ # class may be derived from Network
        self.__class__ = NetworkNoMethods # stops recursion in copy.copy
        net = copy.copy(self) # we make a copy because after returning from this function we can't restore the class
        self.__class__ = oldclass # restore the class of the original, which is now back in its original state
        net._update_schedule = None # remove the problematic element from the copy
        return (unpickle_network, (oldclass, net)) # the unpickle_network function called with arguments oldclass, net restores it as it was

# This class just used as a general 'heap' class - has no methods but can have attributes
class NetworkNoMethods(object):
    pass

def unpickle_network(oldclass, net):
    # See Network.__reduce__ for an explanation, basically the _update_schedule
    # cannot be pickled because it contains instance methods, but it can just be
    # rebuilt.
    net.__class__ = oldclass
    net._build_update_schedule()
    return net


class NetworkOperation(magic.InstanceTracker, ObjectContainer):
    """Callable class for operations that should be called every update step
    
    Typically, you should just use the :func:`network_operation` decorator, but if you
    can't for whatever reason, use this. Note: current implementation only works for
    functions, not any callable object.
    
    **Initialisation:** ::
    
        NetworkOperation(function[,clock])

    If your function takes an argument, the clock will be passed
    as that argument.
    """
    def __init__(self, function, clock=None, when='end'):
        self.clock = guess_clock(clock)
        self.when = when
        self.function = function
        if hasattr(function, 'func_code'):
            self._has_arg = (self.function.func_code.co_argcount==1)

    def __call__(self):
        if self._has_arg:
            self.function(self.clock)
        else:
            self.function()


def network_operation(*args, **kwds):
    """Decorator to make a function into a :class:`NetworkOperation`
    
    A :class:`NetworkOperation` is a callable class which is called every
    time step by the :class:`Network` ``run`` method. Sometimes it is useful
    to just define a function which is to be run every update step. This
    decorator can be used to turn a function into a :class:`NetworkOperation`
    to be added to a :class:`Network` object.
    
    **Example usages**
    
    Operation doesn't need a clock::
    
        @network_operation
        def f():
            ...
        
    Automagically detect clock::
    
        @network_operation
        def f(clock):
            ...
    
    Specify a clock::
    
        @network_operation(specifiedclock)
        def f(clock):
            ...
        
    Specify when the network operation is run (default is ``'end'``)::
    
        @network_operation(when='start')
        def f():
            ...
    
    Then add to a network as follows::
    
        net = Network(f,...)
    """
    # Notes on this decorator:
    # Normally, a decorator comes in two types, with or without arguments. If
    # it has no arguments, e.g.
    #   @decorator
    #   def f():
    #      ...
    # then the decorator function is defined with an argument, and that
    # argument is the function f. In this case, the decorator function
    # returns a new function in place of f.
    #
    # However, you can also define:
    #   @decorator(arg)
    #   def f():
    #      ...
    # in which case the argument to the decorator function is arg, and the
    # decorator function returns a 'function factory', that is a callable
    # object that takes a function as argument and returns a new function.
    #
    # It might be clearer just to note that the first form above is equivalent
    # to:
    #   f = decorator(f)
    # and the second to:
    #   f = decorator(arg)(f)
    #
    # In this case, we're allowing the decorator to be called either with or
    # without an argument, so we have to look at the arguments and determine
    # if it's a function argument (in which case we do the first case above),
    # or if the arguments are arguments to the decorator, in which case we
    # do the second case above.
    #
    # Here, the 'function factory' is the locally defined class
    # do_network_operation, which is a callable object that takes a function
    # as argument and returns a NetworkOperation object.
    class do_network_operation(object):
        def __init__(self, clock=None, when='end'):
            self.clock = clock
            self.when = when
        def __call__(self, f, level=1):
            new_network_operation = NetworkOperation(f, self.clock, self.when)
            # Depending on whether we were called as @network_operation or
            # @network_operation(...) we need different levels, the level is
            # 2 in the first case and 1 in the second case (because in the
            # first case we go originalcaller->network_operation->do_network_operation
            # and in the second case we go originalcaller->do_network_operation
            # at the time when this method is called).
            new_network_operation.set_instance_id(level=level)
            new_network_operation.__name__ = f.__name__
            new_network_operation.__doc__ = f.__doc__
            new_network_operation.__dict__.update(f.__dict__)
            return new_network_operation
    if len(args) == 1 and callable(args[0]):
        # We're in case (1), the user has written:
        # @network_operation
        # def f():
        #    ...
        # and the single argument to the decorator is the function f
        return do_network_operation()(args[0], level=2)
    else:
        # We're in case (2), the user has written:
        # @network_operation(...)
        # def f():
        #    ...
        # and the arguments might be clocks or strings, and may have been
        # called with or without names, so we check both the variable length
        # argument list *args, and the keyword dictionary **kwds, falling
        # back on the default values if nothing is given.
        clk = None
        when = 'end'
        for arg in args:
            if isinstance(arg, Clock):
                clk = arg
            elif isinstance(arg, str):
                when = arg
        for key, val in kwds.iteritems():
            if key == 'clock': clk = val
            if key == 'when': when = val
        return do_network_operation(clock=clk, when=when)
    #raise TypeError, "Decorator must be used as @network_operation or @network_operation(clock)"


class MagicNetwork(Network):
    '''
    Creates a :class:`Network` object from any suitable objects
    
    **Initialised as:** ::
    
        MagicNetwork()
    
    The object returned can then be used just as a regular
    :class:`Network` object. It works by finding any object in
    the ''execution frame'' (i.e. in the same function, script
    or section of module code where the :class:`MagicNetwork` was
    created) derived from :class:`NeuronGroup`, :class:`Connection` or
    :class:`NetworkOperation`.
    
    **Sample usage:** ::
    
        G = NeuronGroup(...)
        C = Connection(...)
        @network_operation
        def f():
            ...
        net = MagicNetwork()
    
    Each of the objects ``G``, ``C`` and ``f`` are added to ``net``.
    
    **Advanced usage:** ::
    
        MagicNetwork([verbose=False[,level=1]])
    
    with arguments:
    
    ``verbose``
        Set to ``True`` to print out a list of objects that were
        added to the network, for debugging purposes.
    ``level``
        Where to find objects. ``level=1`` means look for objects
        where the :class:`MagicNetwork` object was created. The ``level``
        argument says how many steps back in the stack to look.
    '''
    def __init__(self, verbose=False, level=1):
        '''
        Set verbose=False to turn off comments.
        The level variable contains the location of the namespace.
        '''
        (groups, groupnames) = magic.find_instances(NeuronGroup)
        groups = [g for g in groups if g._owner is g]
        groupnames = [gn for g, gn in zip(groups, groupnames) if g._owner is g]
        (connections, connectionnames) = magic.find_instances(Connection)
        (operations, operationnames) = magic.find_instances(NetworkOperation)
        if verbose:
            print "[MagicNetwork] Groups:", groupnames
            print "[MagicNetwork] Connections:", connectionnames
            print "[MagicNetwork] Operations:", operationnames
        # Use set() to discard duplicates
        Network.__init__(self, list(set(groups)), list(set(connections)), list(set(operations)))


def run(duration, threads=1, report=None, report_period=10 * second):
    '''
    Run a network created from any suitable objects that can be found
    
    Arguments:
    
    ``duration``
        the length of time to run the network for.
    ``report``
        How to report progress, the default ``None`` doesn't report the
        progress. Some standard values for ``report``:
        
        ``text``, ``stdout``
            Prints progress to the standard output.
        ``stderr``
            Prints progress to the standard error output stderr.
        ``graphical``, ``tkinter``
            Uses the Tkinter module to show a graphical progress bar,
            this may interfere with any other GUI code you have.
            
        Alternatively, you can provide your own callback function by
        setting ``report`` to be a function ``report(elapsed, complete)``
        of two variables ``elapsed``, the amount of time elapsed in
        seconds, and ``complete`` the proportion of the run duration
        simulated (between 0 and 1). The ``report`` function is
        guaranteed to be called at the end of the run with
        ``complete=1.0`` so this can be used as a condition for
        reporting that the computation is finished.
    ``report_period``
        How often the progress is reported (by default, every 10s).
    
    Works by constructing a :class:`MagicNetwork` object from all the suitable
    objects that could be found (:class:`NeuronGroup`, :class:`Connection`, etc.) and
    then running that network. Not suitable for repeated runs or situations
    in which you need precise control.
    '''
    MagicNetwork(verbose=False, level=2).run(duration, threads=threads,
                                            report=report, report_period=report_period)


def reinit(states=True):
    '''
    Reinitialises any suitable objects that can be found
    
    **Usage:** ::
    
        reinit(states=True)
    
    Works by constructing a :class:`MagicNetwork` object from all the suitable
    objects that could be found (:class:`NeuronGroup`, :class:`Connection`, etc.) and
    then calling ``reinit()`` for each of them. Not suitable for repeated
    runs or situations in which you need precise control.
    
    If ``states=False`` then :class:`NeuronGroup` state variables will not be
    reinitialised.
    '''
    MagicNetwork(verbose=False, level=2).reinit(states=states)


def stop():
    '''
    Globally stops any running network, this is reset the next time a network is run
    '''
    global globally_stopped
    globally_stopped = True

def clear(erase=True, all=False):
    '''
    Clears all Brian objects.
    
    Specifically, it stops all existing Brian objects from being collected by
    :class:`MagicNetwork` (objects created after clearing will still be collected).
    If ``erase`` is ``True`` then it will also delete all data from these objects.
    This is useful in, for example, ``ipython`` which stores persistent references
    to objects in any given session, stopping the data and memory from being freed
    up.  If ``all=True`` then all Brian objects will be cleared. See also
    :func:`forget`.
    '''
    if all is False:
        net = MagicNetwork(level=2)
        objs = net.groups + net.connections + net.operations
    else:
        groups, _ = magic.find_instances(NeuronGroup, all=True)
        connections, _ = magic.find_instances(Connection, all=True)
        operations, _ = magic.find_instances(NetworkOperation, all=True)
        objs = groups+connections+operations
    for o in objs:
        o.set_instance_id(-1)
        if erase:
            for k, v in o.__dict__.iteritems():
                object.__setattr__(o, k, None)
    gc.collect()

def forget(*objs):
    '''
    Forgets the list of objects passed
    
    Forgetting means that :class:`MagicNetwork` will not pick up these objects,
    but all data is retained. You can pass objects or lists of objects. Forgotten
    objects can be recalled with :func:`recall`. See also :func:`clear`.
    '''
    for obj in objs:
        if isinstance(obj, (NeuronGroup, Connection, NetworkOperation)):
            obj._forgotten_instance_id = obj.get_instance_id()
            obj.set_instance_id(-1)
        elif isSequenceType(obj):
            for o in obj:
                forget(o)
        else:
            raise TypeError('Only the following types of objects can be forgotten: NeuronGroup, Connection or NetworkOperation')

def recall(*objs):
    '''
    Recalls previously forgotten objects
    
    See :func:`forget` and :func:`clear`.
    '''
    for obj in objs:
        if isinstance(obj, (NeuronGroup, Connection, NetworkOperation)):
            if hasattr(obj, '_forgotten_instance_id'):
                obj.set_instance_id(obj._forgotten_instance_id)
        elif isSequenceType(obj):
            for o in obj:
                recall(o)
        else:
            raise TypeError('Only the following types of objects can be recalled: NeuronGroup, Connection or NetworkOperation')
