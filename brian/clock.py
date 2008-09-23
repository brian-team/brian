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
"""
Clocks for the simulator
"""

__docformat__ = "restructuredtext en"

__all__ = ['Clock','defaultclock','guess_clock','define_default_clock','reinit_default_clock','get_default_clock']

from inspect import stack
from units import *
from globalprefs import *
import magic
from time import time

# defines and tests the interface, the docstring is considered part of the definition
def _define_and_test_interface(self):
    """
    The :class:`Clock` object
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    
    A :class:`Clock` object is initialised via::
    
        c = Clock(dt=0.1*msecond, t=0*msecond, makedefaultclock=False)
    
    In particular, the following will work and do the same thing::
    
        c = Clock()
        c = Clock(t=0*second)
        c = Clock(dt=0.1*msecond)
        c = Clock(0.1*msecond,0*second)
    
    Setting the ``makedefaultclock=True`` argument sets the newly
    created clock as the default one.

    The default clock
    ~~~~~~~~~~~~~~~~~
    
    The default clock can be found using the :func:`get_default_clock` function,
    and redefined using the :func:`define_default_clock` function, where the
    arguments passed to :func:`define_default_clock` are the same as the
    initialising arguments to the ``Clock(...)`` statement. The default
    clock can be reinitialised by calling :func:`reinit_default_clock`.
    
    A less safe way to access the default clock is to refer directly to
    the variable :data:`defaultclock`. If the default clock has been redefined, this
    won't work.

    The :func:`guess_clock` function
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    The function :func:`guess_clock` should return (in this order of priority):
    
    * The clock passed as an argument, if one was passed
    * A clock defined in the calling function, if there was one
    * The default clock otherwise
    
    If more than one clock was defined in the calling function, it should
    raise a ``TypeError`` exception.
    """
    
    # check that 'defaultclock' default clock exists and starts at t=0
    self.assert_(defaultclock.t<0.001*msecond)
    
    # check that default clock exists and starts at t = 0
    c = guess_clock()
    self.assert_(c.t<0.001*msecond)

    # check that passing no arguments works
    c = Clock()
    
    # check that passing t argument works
    c = Clock(t=1*second)
    self.assert_(c.t>0.9*second)
    
    # check that passing dt argument works
    c = Clock(dt=10*msecond)
    self.assert_(c.dt>9*msecond)
    
    # check that passing t and dt arguments works
    c = Clock(t=2*second, dt=1*msecond)
    self.assert_(c.t>1.9*second)
    self.assert_(0.5*msecond<c.dt<2*msecond)
    
    # check that making this the default clock works
    self.assert_(get_global_preference('defaultclock').dt<9*msecond)
    c = Clock(dt=10*msecond,makedefaultclock=True)
    self.assert_(get_global_preference('defaultclock').dt>9*msecond)

    # check that the other ways of defining a default clock work
    define_default_clock(dt=3*msecond)
    self.assert_(2.9*msecond<get_global_preference('defaultclock').dt<3.1*msecond)
    
    # check that the get_default_clock function works
    self.assert_(2.9*msecond<get_default_clock().dt<3.1*msecond)
    
    # check that passing unnamed arguments in the order dt, t works
    c = Clock(10*msecond,1*second)
    self.assert_(c.t>0.9*second)
    self.assert_(c.dt>9*msecond)
    
    # check that reinit() sets t=0
    c = Clock(t=1*second)
    c.reinit()
    self.assert_(c.t<0.001*second)
    
    # check that tick() works
    c = Clock(t=1*second, dt=1*msecond)
    for i in range(10): c.tick()
    self.assert_(c.t>9*msecond)
    
    # check that reinit_default_clock works
    for i in range(10): get_default_clock().tick()
    reinit_default_clock()
    self.assert_(get_default_clock().t<0.0001*second)
    
    # check that guess_clock passed a clock returns that clock
    self.assert_(0.6*second<guess_clock(Clock(t=0.7*second)).t<0.8*second)
    
    # check that guess_clock passed no clock returns the only clock we've defined in this function so far
    c = Clock()
    self.assert_(guess_clock() is c)
    
    # check that if no clock is defined, guess_clock returns the default clock
    del c
    self.assert_(guess_clock() is get_default_clock())
    
    # check that if two or more clocks are defined, guess_clock raises a TypeError
    c = Clock()
    d = Clock()
    self.assertRaises(TypeError,guess_clock)
    del d
        
    # check that if we have a calling stack, the innermost clock is found
    c = Clock()
    def f(self):
        d = Clock()
        self.assert_(guess_clock() is d)
        del d
        return guess_clock()
    self.assert_(f(self) is c)
    
    # cleanup: reset the default clock to its default state
    set_global_preferences(defaultclock=defaultclock) 

class Clock(magic.InstanceTracker):
    '''
    An object that holds the simulation time and the time step.
    
    Initialisation arguments:
    
    ``dt``
        The time step of the simulation.
    ``t``
        The current time of the clock.
    ``makedefaultclock``
        Set to ``True`` to make this clock the default clock.
        
    **Methods**
    
    .. method:: reinit([t=0*second])
    
        Reinitialises the clock time to zero (or to your
        specified time).
    
    **Attributes**
    
    .. attribute:: t
                   dt
    
        Current time and time step with units.
    
    **Advanced**
    
    *Attributes*
    
    .. attribute:: end
    
        The time at which the current simulation will end,
        set by the :meth:`Network.run` method.
    
    *Methods*
    
    .. method:: tick()
    
        Advances the clock by one time step.
        
    .. method:: set_t(t)
                set_dt(dt)
                set_end(end)
    
        Set the various parameters.
        
    .. method:: get_duration()
    
        The time until the current simulation ends.
        
    .. method:: set_duration(duration)
    
        Set the time until the current simulation ends.
        
    .. method:: still_running()
    
        Returns a ``bool`` to indicate whether the current
        simulation is still running.
    
    For reasons of efficiency, we recommend using the methods
    :meth:`tick`, :meth:`set_duration` and :meth:`still_running`
    (which bypass unit checking internally).
    '''
      
    @check_units(dt=second,t=second)
    def __init__(self,dt=0.1*msecond,t=0*msecond,makedefaultclock=False):
        self._t=float(t)
        self._dt=float(dt)
        self._end=float(t)
        if not exists_global_preference('defaultclock') or makedefaultclock:
            set_global_preferences(defaultclock=self)
    
    @check_units(t=second)
    def reinit(self,t=0*msecond):
        self._t = float(t)
    
    def tick(self):
        self._t+=self._dt
        
    def __repr__(self):
        return 'Clock: t = '+str(self.t)+', dt = '+str(self.dt)
    
    def __str__(self):
        '''
        Returns the current time.
        '''
        return str(self.t)
    
    @check_units(dt=second)
    def set_dt(self,dt):
        self._dt = float(dt)
    
    @check_units(t=second)
    def set_t(self,t):
        self._t = float(t)
        self._end = float(t)

    @check_units(end=second)
    def set_end(self,end):
        """Sets the end-point for the clock
        """
        self._end = float(end)
    
    # Clock object internally stores floats, but these properties
    # return quantities
    t=property(fget=lambda self:self._t*second,fset=set_t)
    dt=property(fget=lambda self:self._dt*second,fset=set_dt)
    end=property(fget=lambda self:self._end*second,fset=set_end)
    
    @check_units(duration=second)
    def set_duration(self,duration):
        """Sets the duration of the clock
        """
        self._end = self._t + float(duration)
    
    def get_duration(self):
        return self.end-self.t
    
    def still_running(self):
        """Checks if the clock is still running
        """
        return self._t < self._end

def guess_clock(clock=None):
    '''
    Tries to guess the clock from global and local namespaces
    from the caller.
    Selects the most local clock.
    Raises an error if several clocks coexist in the same namespace.
    If a non-None clock is passed, then it is returned (simplifies the code).
    '''
    if clock:
        return clock
    # Get variables from the stack
    (clocks,clocknames) = magic.find_instances(Clock)
    if len(clocks)>1: # several clocks: ambiguous
        # What type of error?
        raise TypeError,"Clock is ambiguous. Please specify it explicitly."
    if len(clocks)==1:
        return clocks[0]
    # Fall back on default clock
    if exists_global_preference('defaultclock'): return get_global_preference('defaultclock')
    # No clock found
    raise TypeError,"No clock found. Please define a clock."

# Do not track the default clock    
class DefaultClock(Clock):
    @staticmethod
    def _track_instances(): return False    
defaultclock = DefaultClock(dt=0.1*msecond)
define_global_preference('defaultclock','Clock(dt=0.1*msecond)',
                           desc = """
                                  The default clock to use if none is provided or defined
                                  in any enclosing scope.
                                  """)

def define_default_clock(**kwds):
    '''
    Create a new default clock
    
    Uses the keywords of the :class:`Clock` initialiser.
    
    Sample usage::
    
        define_default_clock(dt=1*ms)
    '''
    kwds['makedefaultclock']=True
    newdefaultclock = Clock(**kwds)

def reinit_default_clock(t=0*msecond):
    '''
    Reinitialise the default clock (to zero or a specified time)
    '''
    get_default_clock().reinit(t)

def get_default_clock():
    '''
    Returns the default clock object.
    '''
    return get_global_preference('defaultclock')

if __name__=='__main__':
    print id(guess_clock()), id(defaultclock)