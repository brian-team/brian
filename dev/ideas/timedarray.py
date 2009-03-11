from brian.clock import *
from brian.network import *
import numpy
import pylab

class TimedArray(numpy.ndarray):
    def __new__(subtype, arr, times=None, clock=None):
        return numpy.array(arr, copy=False).view(subtype)
    def __init__(self, arr, times=None, clock=None):
        if (times is None and clock is None) or (times is not None and clock is not None):
            raise ValueError('Specify times or clock but not both.')
        if clock is not None:
            times = clock.t+arange(len(arr))*clock.dt
        self.times = times
    def plot(self, *args, **kwds):
        if self.size>self.times.size and len(self.shape)==2:
            for i in xrange(self.shape[1]):
                kwds['label'] = str(i)
                self[:, i].plot(*args, **kwds)
        else:
            pylab.plot(self.times, self, *args, **kwds)
    def __getitem__(self, item):
        x = numpy.ndarray.__getitem__(self, item)
        if isinstance(item, slice):
            return TimedArray(x, self.times[item])
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
        except IndexError:
            pass
        raise IndexError('Not sure what is going on.')
    def __getslice__(self, start, end):
        x = numpy.ndarray.__getslice__(self, start, end)
        return TimedArray(x, self.times[start:end])
    def __array_finalize__(self, orig):
        try:
            self.times = orig.times
        except AttributeError:
            pass
        return self

# TODO: idea is that we should be able to set a NeuronGroup parameter
# using a TimedArray, and it will set the values according to the times.
# (Could do interpolation?) Implement this as a NetworkOperation object,
# but also have a function that adds it to the contained_objects of the
# group it is applied to. (Rather add it as a method to NeuronGroup?)
# Efficient algorithm would be to implement a
# reinit method that restarts a generator that yields values according
# to times.

#class TimedArraySetter(NetworkOperation):
#    def __init__(self, group, var, array, times=None, dt=None):

#def set_group_var_by_timed_array(group, var, array, times=None, dt=None): 

if __name__=='__main__':
    from brian import *
    x = array([t*(arange(10)+randn(10)) for t in arange(100)/100.])
    y = TimedArray(x, clock=Clock(dt=1*second/100))
    z = y[1,1:5]
    print z.shape
    print asarray(z)
    print z.times
    #z.plot()
    #show()
#    for i in range(10):
#        y[10:30,i].plot()
    y[10:30,:].plot()
    legend()
    show()