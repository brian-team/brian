from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

#x = Sound(randn(15,2), samplerate=100*Hz)
#
#print x
#print x.buffer_fetch(0, 10)
#print x.buffer_fetch(5, 15)
#print x.buffer_fetch(10, 20)

#y = Sound(0.1*randn(200,2), samplerate=100*Hz)
#
#x = x+y
#x = x.shift(500*ms)
#x = Sound.sequence((x, x))
#print x.intensity('peak')
#x = x.repeat(2)
#x.copy_from(y)

x = Sound.tone(500*Hz, 500*ms)
y = Sound.tone(5000*Hz, 500*ms)
x, y = y, x
#z = Sound(array([x.flatten(), y.flatten()]).T, samplerate=x.samplerate)
z = Sound((x, y), samplerate=x.samplerate)
z.play(True, True)
x=z

#x = Sound([lambda t:sin(2*pi*t*50),
#           lambda t:cos(2*pi*t*100)], samplerate=10000*Hz, duration=1*second)

#x = Sound(randn(100,2), samplerate=100*Hz).ramped(duration=200*ms)
#x = Sound.tone(500*Hz, 500*ms)

#for i in xrange(x.nchannels):
#    plot(x.times, x.channel(i))
#show()

#x = Sound(randn(44100,2), samplerate=44100*Hz)
#x.left.spectrogram()
#show()