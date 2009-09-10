from pylab import *
import time

ion()

tstart = time.time()               # for profiling
#x = arange(0,2*pi,0.01)            # x-array
x = []
line, = plot(x,sin(x))
axis([0,2*pi,-1,1])
for i in arange(1,200):
    x = arange(0,2*pi*(i/200.),0.01)
    line.set_xdata(x)
    line.set_ydata(sin(x+i/10.0))  # update the data
    draw()                         # redraw the canvas

print 'FPS:' , 200/(time.time()-tstart)
