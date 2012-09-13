#!/usr/bin/env python
'''
Example of a 2D stimulus, see the
`complete description <http://neuralensemble.org/cookbook/wiki/Brian/StimulusArrayGroup>`__
at the Brian Cookbook.
'''

from brian import *
import scipy.ndimage as im

__all__ = ['bar', 'StimulusArrayGroup']

def bar(width, height, thickness, angle):
    '''
    An array of given dimensions with a bar of given thickness and angle
    '''
    stimulus = zeros((width, height))
    stimulus[:, int(height / 2. - thickness / 2.):int(height / 2. + thickness / 2.)] = 1.
    stimulus = im.rotate(stimulus, angle, reshape=False)
    return stimulus


class StimulusArrayGroup(PoissonGroup):
    '''
    A group of neurons which fire with a given stimulus at a given rate
    
    The argument ``stimulus`` should be a 2D array with values between 0 and 1.
    The point in the stimulus array at position (y,x) will correspond to the
    neuron with index i=y*width+x. This neuron will fire Poisson spikes at
    ``rate*stimulus[y,x]`` Hz. The stimulus will start at time ``onset``
    for ``duration``.
    '''
    def __init__(self, stimulus, rate, onset, duration):
        height, width = stimulus.shape
        stim = stimulus.ravel()*rate
        self.stimulus = stim
        def stimfunc(t):
            if onset < t < (onset + duration):
                return stim
            else:
                return 0. * Hz
        PoissonGroup.__init__(self, width * height, stimfunc)

if __name__ == '__main__':
    import pylab
    subplot(121)
    stim = bar(100, 100, 10, 90) * 0.9 + 0.1
    pylab.imshow(stim, origin='lower')
    pylab.gray()
    G = StimulusArrayGroup(stim, 50 * Hz, 100 * ms, 100 * ms)
    M = SpikeMonitor(G)
    run(300 * ms)
    subplot(122)
    raster_plot(M)
    axis(xmin=0, xmax=300)
    show()
