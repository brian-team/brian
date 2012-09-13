#!/usr/bin/env python
'''
Example of basic use and manipulation of sounds with Brian hears.
'''
from brian import *
from brian.hears import *

sound1 = tone(1*kHz, 1*second)
sound2 = whitenoise(1*second)

sound = sound1+sound2
sound = sound.ramp()

# Comment this line out if you don't have pygame installed
sound.play()

# The first 20ms of the sound
startsound = sound[:20*ms]

subplot(121)
plot(startsound.times, startsound)
subplot(122)
sound.spectrogram()
show()
