from brian import *
from brian.library.electrophysiology import Lp_compensate
import numpy

I = numpy.load("current.npy")
Vraw = numpy.load("trace.npy")

Vcomp, params = Lp_compensate(I, Vraw, .1*ms)

subplot(211)
plot(I, 'k')

subplot(212)
plot(Vraw, 'k')
plot(Vcomp, 'r')

show()
