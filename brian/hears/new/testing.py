from brian import *
from brian.hears import *

x = Sound(randn(100,2), rate=100*Hz)
y = Sound(0.1*randn(200,3), rate=100*Hz)

x = x+y

plot(x.times, x[:, 0])
plot(x.times, x[:, 1])
show()
