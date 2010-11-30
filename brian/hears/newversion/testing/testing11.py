from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

x = Sound(randn(15, 1), samplerate=100*Hz)
fb = DoNothingFilterbank(x)
fb.maximum_buffer_size = 5
y = fb.buffer_fetch(0, 15)

print amax(abs(y-asarray(x)))

#plot(y)
#plot(x)
#show()
