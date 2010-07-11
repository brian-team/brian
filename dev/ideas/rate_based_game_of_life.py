'''
This demonstrates the 'Game of Life' example from DANA, integrated into
Brian's framework.
'''

from brian import *
from scipy.signal import convolve2d
from pylab import gray

width = 50
N = width*width
steps = 64

duration = steps*defaultclock.dt

kernel = array([[1,1,1],
                [1,0,1],
                [1,1,1]])

G = NeuronGroup(N, 'alive:1')

G.alive = randint(2, size=N)
alive = reshape(G.alive, (width, width))

plotshape = int(ceil(sqrt(steps)))

plotnum = 1
@network_operation
def update():
    global plotnum
    neighbs = convolve2d(alive, kernel, mode='same')
    alive[:] = maximum(0, 1.0-(neighbs<1.5)-(neighbs>3.5)-(neighbs<2.5)*(1-alive))
    if plotnum<=plotshape*plotshape:
        subplot(plotshape, plotshape, plotnum)
        imshow(alive.copy(), interpolation='nearest')
        gray()
    plotnum += 1

run(10*ms)

show()
