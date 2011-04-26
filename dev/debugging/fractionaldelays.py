from brian import *
from brian.hears import *

sound = whitenoise(10*ms)

delays = linspace(-1*ms, 1*ms, 1000)

fd = FractionalDelay(sound, delays)
fd2 = FractionalDelay(sound, array(delays*sound.samplerate, dtype=int)/sound.samplerate)

I = fd.process()
I2 = fd2.process()

imshow(I, interpolation='nearest', aspect='auto')
figure()
imshow(I2, interpolation='nearest', aspect='auto')
show()
