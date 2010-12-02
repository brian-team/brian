from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

x1 = Sound((whitenoise(100*ms), whitenoise(100*ms)))
x1.level = (0*dB, 10*dB)
x2 = Sound((whitenoise(100*ms), whitenoise(100*ms)))
x2.level = (20*dB, 30*dB)
x3 = Sound((whitenoise(100*ms), whitenoise(100*ms)))
x3.level = (40*dB, 50*dB)
x4 = Sound((whitenoise(100*ms), whitenoise(100*ms)))
x4.level = (60*dB, 70*dB)
x5 = Sound((whitenoise(100*ms), whitenoise(100*ms)))
x5.level = (80*dB, 90*dB)

fb1 = RestructureFilterbank((x1, x2), numrepeat=3, type='serial')
fb2 = RestructureFilterbank((x3, x4), type='interleave', numtile=3)
fb3 = RestructureFilterbank((fb1, fb2), type='interleave')
rep = Repeat(x1, 3)
fb4 = Interleave(Join(rep,
                      Repeat(x2, 3)),
                 Tile(Interleave(x3, x4), 3))

print fb1.indexmapping
print fb2.indexmapping
print fb3.indexmapping
print fb4.indexmapping

names = ['x1', 'x2', 'x3', 'x4', 'x5', 'fb1', 'fb2', 'fb3', 'fb4']
objids = [eval('id('+name+')') for name in names]
namefromid = dict(zip(objids, names))

for s in fb3.source:
    print namefromid[id(s)]
print '---'
for s in fb4.source:
    print namefromid[id(s)]
    
y = Sound(fb3.buffer_fetch(0, len(x1)))

print y.level

rep.source = x5
fb4.buffer_init()
print Sound(fb4.buffer_fetch(0, len(x1))).level

for s in fb4.source:
    print namefromid[id(s)]
