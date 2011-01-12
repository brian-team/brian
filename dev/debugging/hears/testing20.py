from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

x = whitenoise(10*ms)

fb = GammatoneFilterbank(x, [10*Hz, 20*Hz])

fb2 = Repeat(fb, 3)

for k, v in fb.__dict__.iteritems():
    if 'buffer' in k and not isinstance(v, ndarray):
        print k, ':', v

print '***'

fb.buffer_fetch(0, len(x))

for k, v in fb.__dict__.iteritems():
    if 'buffer' in k and not isinstance(v, ndarray):
        print k, ':', v

fb2.buffer_fetch(0, len(x))
