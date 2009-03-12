from clock import *
from network import *
import numpy
import pylab
import warnings

__all__ = ['TimedArray', 'TimedArraySetter', 'set_group_var_by_array']

class TimedArray(numpy.ndarray):
    '''
    An array which also stores times.
    
    Brief notes (unfinished):
    
    The first index of the array must be the time index. Shapes in mind
    are (T,) and (T, N) for T the number of time steps and N the number
    of neurons.
    
    Slicing operations are supported to some extent but not entirely.
    
    The __call__ operation requires the times to be set by a clock rather
    than an array of times, and operated on versions of the array will not
    have this property and so cannot be called. (But they can be plotted.)
    '''
    def __new__(subtype, arr, times=None, clock=None):
        return numpy.array(arr, copy=False).view(subtype)
    def __array_finalize__(self, orig):
        try:
            self.times = orig.times
        except AttributeError:
            pass
        return self
    def __init__(self, arr, times=None, clock=None):
        if times is not None and clock is not None:
            raise ValueError('Specify times or clock but not both.')
        if times is None and clock is None:
            clock = guess_clock(clock)
        self.clock = clock
        if clock is not None:
            self._t_init = clock._t
            self._dt = clock._dt
            times = clock._t+numpy.arange(len(arr))*clock._dt
        self.times = times
    def __getitem__(self, item):
        x = numpy.ndarray.__getitem__(self, item)
        if isinstance(item, slice):
            return TimedArray(x, self.times[item])
        if isinstance(item, int):
            return TimedArray(x, self.times[item:item+1])
        try:
            times = self.times[item]
            if not isinstance(times, numpy.ndarray):
                times = numpy.array([times])
            return TimedArray(x, times)
        except IndexError:
            pass
        try:
            item0 = item[0]
            times = self.times[item0]
            if not isinstance(times, numpy.ndarray):
                times = numpy.array([times])
            return TimedArray(x, times)
        except TypeError:
            pass
        raise IndexError('Not sure what is going on.')
    def __getslice__(self, start, end):
        x = numpy.ndarray.__getslice__(self, start, end)
        return TimedArray(x, self.times[start:end])
    def plot(self, *args, **kwds):
        if self.size>self.times.size and len(self.shape)==2:
            for i in xrange(self.shape[1]):
                kwds['label'] = str(i)
                self[:, i].plot(*args, **kwds)
        else:
            pylab.plot(self.times, self, *args, **kwds)
    def __call__(self, t):
        if self.clock is None:
            raise ValueError('Can only call timed arrays if they are based on a clock.')
        else:
            if isinstance(t, numpy.ndarray):
                # Normally would not support numpy.ndarray except Brian uses it for
                # the value of t in equations at the moment (this may change). So
                # we just the first value because when used by Brian all values are
                # the same.
                t = t[0] 
            else:
                t = float(t)
            t = int((t-self._t_init)/self._dt)
            if t<0: t=0
            if t>=len(self.times): t=len(self.times)-1
            return numpy.asarray(self)[t]

class TimedArraySetter(NetworkOperation):
    def __init__(self, group, var, arr, times=None, clock=None, when='start'):
        self.clock = guess_clock(clock)
        self.when = when
        self.group = group
        self.var = var
        if not isinstance(arr, TimedArray):
            arr = TimedArray(arr, times=times, clock=clock)
        self.arr = arr
    def __call__(self):
        # Could write an efficient implementation of this that works even if
        # the array doesn't have an associated clock. Have a reinit method
        # and a generator that steps through updating to the next index when
        # necessary.
        self.group.state_(self.var)[:] = self.arr(self.clock._t)

def set_group_var_by_array(group, var, arr, times=None, clock=None):
    array_setter = TimedArraySetter(group, var, arr, times=times, clock=clock)
    group.contained_objects.append(array_setter) 
