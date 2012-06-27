from brian import *
from brian.hears import *

mono_sound = tone(500*Hz, duration=1*second)
stereo_sound = tone(500*Hz, duration=1*second, nchannels=2)

# this works
assert(mono_sound.shifted(5*ms).shape[0] == stereo_sound.shifted(5*ms).shape[0])

# this does not (the stereo sound is transposed, i.e. now has two samples and
# 44100 channels)
assert(mono_sound.shifted(5*ms, fractional=True).shape[0] ==
       stereo_sound.shifted(5*ms, fractional=True).shape[0])