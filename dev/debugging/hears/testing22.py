from brian import *
from brian.hears import *

x = whitenoise(100*ms)
fb = Gammatone(x, [1*kHz, 1.01*kHz])

#fb.duration = 2*second

print fb.duration

nsamples = x.nsamples

def sum_of_squares(input, running_sum_of_squares):
    return running_sum_of_squares+sum(input**2, axis=0)
rms = sqrt(fb.process(sum_of_squares)/nsamples)

print rms

y = fb.process()
y = fb.process()

print sqrt(sum(y**2, axis=0)/nsamples)

plot(y)
show()
