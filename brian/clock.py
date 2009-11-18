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

__all__ = ['Clock','defaultclock','guess_clock','define_default_clock','reinit_default_clock','get_default_clock',
           'EventClock', 'RegularClock']

from inspect import stack
from units import *
from globalprefs import *
import magic
from time import time

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

    @check_units(start=second)
    def set_start(self,start):
        """Sets the start-point for the clock
        """
        self._start = float(start)
    
    # Clock object internally stores floats, but these properties
    # return quantities
    if isinstance(second,Quantity):
        t=property(fget=lambda self:Quantity.with_dimensions(self._t,second.dim),fset=set_t)
        dt=property(fget=lambda self:Quantity.with_dimensions(self._dt,second.dim),fset=set_dt)
        end=property(fget=lambda self:Quantity.with_dimensions(self._end,second.dim),fset=set_end)
        start=property(fget=lambda self:Quantity.with_dimensions(self._start,second.dim),fset=set_start)
    else:
        t=property(fget=lambda self:self._t,fset=set_t)
        dt=property(fget=lambda self:self._dt,fset=set_dt)
        end=property(fget=lambda self:self._end,fset=set_end)
        start=property(fget=lambda self:self._start,fset=set_start)
    
    @check_units(duration=second)
    def set_duration(self,duration):
        """Sets the duration of the clock
        """
        self._start = self._t
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

class EventClock(Clock):
    '''
    Clock that is used for events.
    
    Works the same as a :class:`Clock` except that it is never guessed as a clock to
    use by :class:`NeuronGroup`, etc. These clocks can be used to make multiple clock
    simulations without causing ambiguous clock problems.
    '''
    @staticmethod
    def _track_instances(): return False    

class RegularClock(Clock):
    '''
    Clock that always ticks to integer multiples of dt
    
    Works the same as a :class:`Clock`, except that underlying times are stored as
    integers rather than floats, so it doesn't drift over time due to accumulated
    tiny errors in floating point arithmetic. The initialiser
    has one extra parameter, ``offset``. Clock times will be of the form
    ``i*dt+offset``. It is usually better to have a small offset to ensure that
    ``t`` is always in the interval ``[i*dt, (i+1)*dt)``.
    '''
    @check_units(dt=second,t=second)
    def __init__(self, dt=0.1*msecond, t=0*msecond, offset=1e-15*second, makedefaultclock=False):
        self._gridoffset = float(gridoffset)
        self.__t = int(t/dt)
        self.__dt = 1
        self._dt = float(dt)
        self.__end = self.__t
        if not exists_global_preference('defaultclock') or makedefaultclock:
            set_global_preferences(defaultclock=self)
    @check_units(t=second)
    def reinit(self,t=0*msecond):
        self.__t = int(float(t)/self._dt)
    def tick(self):
        self.__t += self.__dt
    @check_units(t=second)
    def set_t(self,t):
        self.__t = int(float(t)/self._dt)
        self.__end = int(float(t)/self._dt)
    @check_units(dt=second)
    def set_dt(self,dt):
        self._dt = float(dt)
    @check_units(end=second)
    def set_end(self,end):
        self.__end = int(float(end)/self._dt)
    @check_units(start=second)
    def set_start(self,start):
        self.__start = int(float(start)/self._dt)
    # Regular clock uses integers, but lots of Brian code extracts _t and _dt
    # directly from the clock, so these should be implemented directly
    _t = property(fget=lambda self:self.__t*self._dt+self._gridoffset)
    _end = property(fget=lambda self:self.__end*self._dt+self._gridoffset)
    _start = property(fget=lambda self:self.__start*self._dt)
    # Clock object internally stores floats, but these properties
    # return quantities
    if isinstance(second,Quantity):
        t=property(fget=lambda self:Quantity.with_dimensions(self._t,second.dim),fset=set_t)
        dt=property(fget=lambda self:Quantity.with_dimensions(self._dt,second.dim),fset=set_dt)
        end=property(fget=lambda self:Quantity.with_dimensions(self._end,second.dim),fset=set_end)
        start=property(fget=lambda self:Quantity.with_dimensions(self._start,second.dim),fset=set_start)
    else:
        t=property(fget=lambda self:self._t,fset=set_t)
        dt=property(fget=lambda self:self._dt,fset=set_dt)
        end=property(fget=lambda self:self._end,fset=set_end)
        start=property(fget=lambda self:self._start,fset=set_start)
    @check_units(duration=second)
    def set_duration(self,duration):
        self.__start = self.__t
        self.__end = self.__t + int(float(duration)/self._dt)            
    def get_duration(self):
        return self.end-self.t
    def still_running(self):
        return self.__t < self.__end


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