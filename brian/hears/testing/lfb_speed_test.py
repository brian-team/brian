from brian import *
set_global_preferences(usenewbrianhears=True,
                       #useweave=False,
                       )
from brian.hears import *
from scipy.signal import lfilter
import time

N = 1000
duration = 10240
bufsize = 1024

x = Sound(randn(duration, 1), samplerate=44100*Hz)

fb = GammatoneFilterbank(x, erbspace(20*Hz, 20*kHz, N))

if 0:
    # Doing it with lfilter channel by channel
    b, a = fb.filt_b, fb.filt_a
    n, m, p = b.shape
    start = time.time()
    for i in xrange(n):
        for k in xrange(p):
            lfilter(b[i,:,k], a[i,:,k], x)
    print time.time()-start

fb.buffer_init()
start = time.time()
for i in xrange(0, duration, bufsize):
    fb.buffer_fetch(i, i+bufsize)
end = time.time()

print end-start
