from clock import *
from network import *
import neurongroup
from units import second, check_units
import numpy
import warnings
try:
    import pylab
except:
    warnings.warn("Couldn't import pylab.")

__all__ = ['TimedArray', 'TimedArraySetter', 'set_group_var_by_array']


class TimedArray(numpy.ndarray):
    '''
    An array where each value has an associated time.
    
    Initialisation arguments:
    
    ``arr``
        The values of the array. The first index is the time index. Any
        array shape works in principle, but only 1D/2D arrays are
        supported (other shapes may work, but may not). The idea is to,
        have the shapes (T,) or (T, N) for T the number of time steps and
        N the number of neurons.
    ``times``
        A 1D array of times whose length should be the same as the first
        dimension of ``arr``. Usually it is preferable to specify a
        clock rather than an array of times, but this doesn't work in
        the case where the time intervals are not fixed.
    ``clock``
        Specify the times corresponding to array values by a clock. The
        ``t`` attribute of the clock is the time of the first value
        in the array, and the time interval is the ``dt`` attribute of
        the clock. If neither ``times`` nor ``clock`` is specified, a
        clock will be guessed in the usual way (see :class:`Clock`).
    ``start, dt``
        Rather than specifying a clock, you can specify the start time
        and time interval explicitly. Technically, this is useful
        because it doesn't create a :class:`Clock` object which can
        lead to ambiguity about which clock is the default. If dt is
        specified and start is not, start is assumed to be 0.
        
    Note that if the clock, or start time and dt, of the array should be the
    default clock values, then you should not specify clock, start or dt (see
    Technical notes below).
    
    Arbitrary slicing of the array is supported, but the clock will only
    be preserved where the intervals can be guaranteed to be fixed, that
    is except for the case where lists or numpy arrays are used on the
    time index.

    Timed arrays can be called as if they were a function of time if the
    array times are based on a clock (but not if the array times are
    arbitrary as the look up costs would be excessive). If ``x(t)`` is called
    where ``times[i]<=t<times[i]+dt`` for some index i then ``x(t)`` will
    have the value ``x[i]``. You can also call ``x(t)`` with ``t`` a 1D array.
    If x is 1D then ``x(t)[i]=x(t[i])``, if x is 2D then ``x(t)[i]=x(t[i])[i]``.
    
    Has one method:
    
    .. method: plot(*args, **kwds)
    
        Plots the values on the y axis and times on the x axis. If the array
        is 1D then this is a single plot, if it is 2D then there will be one
        plot for each second index. 3D or greater arrays are not supported.
        The args and keywords are passed to matplotlib's plot() command. In
        the 2D case, each plot is labelled with the second index.
    
    See also :class:`TimedArraySetter`, :func:`set_group_var_by_array` and
    :class:`NeuronGroup`.
    
    **Technical notes**
    
    Note that specifying a new clock, or values of start and dt, will mean
    that if you use this :class:`TimedArray` to set the value of a
    :class:`NeuronGroup` variable, it will be updated on the schedule of this
    clock, which can (due to floating point errors) induce some timing problems.
    This rarely happens, but if an occasional inaccuracy of order dt might
    conceivably be critical for your simulation, you should use
    :class:`RegularClock` objects instead of :class:`Clock` objects.
    '''
    def __new__(subtype, arr, times=None, clock=None, start=None, dt=None):
        # All numpy.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        return numpy.array(arr, copy=False).view(subtype)

    def __array_finalize__(self, orig):
        # This is called each time a new TimedArray object is created from
        # an old one, we just copy across the clock attribs here because
        # when a new one is made from an old one, the times will be the
        # same.
        try:
            self.times = orig.times
            self.clock = orig.clock
            self._t_init = orig._t_init
            self._dt = orig._dt
        except AttributeError:
            pass
        return self

    @check_units(start=second, dt=second)
    def __init__(self, arr, times=None, clock=None, start=None, dt=None):
        # Mostly this is straightforward, the point about having
        # times and clock separate is that you don't have to limit
        # yourself to fixed time intervals, although you usually
        # will do that (and some things will rely on this, such
        # as the __call__ method).
        if start is not None or dt is not None:
            if start is None:
                start = 0 * second
            if clock is not None:
                raise ValueError('Specify start and dt or clock, but not both.')
            clock = Clock(t=start, dt=dt)
        if times is not None and clock is not None:
            raise ValueError('Specify times or clock but not both.')
        if times is None and clock is None:
            clock = guess_clock(clock)
            self.guessed_clock = True
        else:
            self.guessed_clock = False
        self.clock = clock
        if clock is not None:
            self._t_init = int(clock._t / clock._dt) * clock._dt
            self._dt = clock._dt
            times = clock._t + numpy.arange(len(arr)) * clock._dt
        else:
            self._t_init = None
            self._dt = None
        self.times = times

    # __reduce__ and __setstate__ are needed for correct pickling and unpickling    
    def __reduce__(self):
        # numpy's reduce function returns a tuple, the third element contains
        # the state that will be fed into the __setstate__ method
        nd_reduce = list(numpy.ndarray.__reduce__(self))
         
        timedarray_state = (self.times, self.clock, self.guessed_clock,
                            self._t_init, self._dt)
        # Return a tuple where the third element contains a combination of the
        # ndarray state and TimedArray's state        
        return (nd_reduce[0], nd_reduce[1], (nd_reduce[2], timedarray_state))
    
    def __setstate__(self,state):
        nd_state, timedarray_state = state
        
        # Restore ndarray's state
        numpy.ndarray.__setstate__(self, nd_state)
        
        # Restore TimedArray's state
        times, clock, guessed_clock, _t_init, _dt = timedarray_state
        self.times = times
        self.clock = clock
        self.guessed_clock = guessed_clock
        self._t_init = _t_init
        self._dt = _dt 

    def __getitem__(self, item):
        # __getitem__ can deal with all sorts of indexing, we consider the
        # following types specially for an array x
        #   - x[a:b:c], if x has a clock then the slice can have a clock too
        #   - x[integer], no clock for this because it's just one time value
        #   - x[a, b, ...] with a a slice, this can have a clock based on a
        # For the remaining cases, we assume that we cannot define a clock for
        # the sliced object, e.g. if x[a,b,...] with a, b numpy arrays.

        # The values are the same as the numpy array version of __getitem__ in
        # all cases
        x = numpy.ndarray.__getitem__(self, item)
        if isinstance(item, slice):
            newtimes = self.times[item]
            if self.clock is not None and len(newtimes) > 1:
                newdt = newtimes[1] - newtimes[0]
                newclock = Clock(t=newtimes[0] * second, dt=newdt * second)
                return TimedArray(x, clock=newclock)
            else:
                return TimedArray(x, self.times[item])
        if isinstance(item, int):
            return TimedArray(x, self.times[item:item + 1])
        if isinstance(item, tuple):
            item0 = item[0]
            times = self.times[item0]
            if isinstance(item0, slice) and self.clock is not None and hasattr(times, '__len__') and len(times) > 1:
                newdt = times[1] - times[0]
                newclock = Clock(t=times[0] * second, dt=newdt * second)
                return TimedArray(x, clock=newclock)
            if not isinstance(times, numpy.ndarray):
                times = numpy.array([times])
            return TimedArray(x, times)
        times = self.times[item]
        if not isinstance(times, numpy.ndarray):
            times = numpy.array([times])
        return TimedArray(x, times)

    def __getslice__(self, start, end):
        # Just use __getitem__ for this (it's been deprecated since Python 2.0
        # but you need to implement it because the base class does)
        return self.__getitem__(slice(start, end))

    def plot(self, *args, **kwds):
        if self.size > self.times.size and len(self.shape) == 2:
            for i in xrange(self.shape[1]):
                kwds['label'] = str(i)
                self[:, i].plot(*args, **kwds)
        else:
            pylab.plot(self.times, self, *args, **kwds)

    def __call__(self, t):
        if self.clock is None:
            raise ValueError('Can only call timed arrays if they are based on a clock.')
        else:
            if isinstance(t, (list, tuple)):
                t = numpy.array(t)
            if isinstance(t, neurongroup.TArray):
                # In this case, we know that t = ones(N)*t so we just use the first value
                t = t[0]
            elif isinstance(t, numpy.ndarray):
                if len(self.shape) > 2:
                    raise ValueError('Calling TimedArray with array valued t only supported for 1D or 2D TimedArray.')
                if len(self.shape) == 2 and len(t) != self.shape[1]:
                    raise ValueError('Calling TimedArray with array valued t on 2D TimedArray requires len(t)=arr.shape[1]')
                t = numpy.array(numpy.rint((t - self._t_init) / self._dt), dtype=int)
                t[t < 0] = 0
                t[t >= len(self.times)] = len(self.times) - 1
                if len(self.shape) == 1:
                    return numpy.asarray(self)[t]
                return numpy.asarray(self)[t, numpy.arange(len(t))]
            t = float(t)
            ot = t
            t = int(numpy.rint((t - self._t_init) / self._dt))
            if t < 0: t = 0
            if t >= len(self.times): t = len(self.times) - 1
            return numpy.asarray(self)[t]


class TimedArraySetter(NetworkOperation):
    '''
    Sets NeuronGroup values with a TimedArray.
    
    At the beginning of each update step, this object will set the
    values of a given state variable of a group with the value from
    the array corresponding to the current simulation time.
    
    Initialisation arguments:
    
    ``group``
        The :class:`NeuronGroup` to which the variable belongs.
    ``var``
        The name or index of the state variable in the group.
    ``arr``
        The array of values used to set the variable in the group.
        Can be an array or a :class:`TimedArray`. If it is an array,
        you should specify the ``times`` or ``clock`` arguments, or
        leave them blank to use the default clock.
    ``times``
        Times corresponding to the array values, see :class:`TimedArray`
        for more details.
    ``clock``
        The clock for the :class:`NetworkOperation`. If none is specified,
        use the group's clock. If ``arr`` is not a :class:`TimedArray`
        then this clock will be used to initialise it too.
    ``start, dt``
        Can specify these instead of a clock (see :class:`TimedArray` for
        details).
    ``when``
        The standard :class:`NetworkOperation` ``when`` keyword, although
        note that the default value is 'start'.
    '''
    @check_units(start=second, dt=second)
    def __init__(self, group, var, arr, times=None, clock=None, start=None, dt=None, when='start'):
        if clock is None:
            if isinstance(arr, TimedArray) and not arr.clock is None:
                self.clock = clock = arr.clock
            else:
                self.clock = clock = group.clock
        else:
            self.clock = clock
        self.when = when
        self.group = group
        self.var = var
        if not isinstance(arr, TimedArray):
            arr = TimedArray(arr, times=times, clock=clock, start=start, dt=dt)
        self.arr = arr
        self.reinit()

    def __call__(self):
        if self.arr.clock is None:
            # in this case, the time intervals need not be fixed so we
            # have to step through the array until we find the appropriate
            # one
            tcur = self.clock._t
            while True:
                if self._cur_i == len(self.arr.times) - 1:
                    self.group.state_(self.var)[:] = self.arr[self._cur_i]
                    return
                ti_next = self.arr.times[self._cur_i + 1]
                if ti_next > tcur:
                    self.group.state_(self.var)[:] = self.arr[self._cur_i]
                    return
                self._cur_i += 1
        else:
            self.group.state_(self.var)[:] = self.arr(self.clock._t)

    def reinit(self):
        if self.arr.clock is None:
            self._cur_i = 0

@check_units(start=second, dt=second)
def set_group_var_by_array(group, var, arr, times=None, clock=None, start=None, dt=None):
    '''
    Sets NeuronGroup values with a TimedArray.
    
    Creates a :class:`TimedArraySetter`, see that class for details.
    '''
    array_setter = TimedArraySetter(group, var, arr, times=times, clock=clock, start=start, dt=dt)
    group._owner.contained_objects.append(array_setter)
