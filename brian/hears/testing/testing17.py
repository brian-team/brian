from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

hrtfdb = IRCAM_LISTEN(r'F:\HRTF\IRCAM', compensated=True)
hrtfset = hrtfdb.load_subject(1002)

print amin(hrtfset.coordinates['elev'])

print len(hrtfset)

hrtfset = hrtfset.subset(lambda elev:abs(elev)<=15)

print len(hrtfset)

hrtf = hrtfset(azim=30, elev=15)

plot(hrtf.left)
plot(hrtf.right)
show()
