from time import time
#http://amtoolbox.sourceforge.net/doc/filters/gammatone.php
from brian import *
set_global_preferences(useweave=True)
from scipy.io import loadmat,savemat

from brian.hears import *
#from zilany import *


simulation_duration = 1*ms
set_default_samplerate(100*kHz)
#sound = whitenoise(simulation_duration)
file="/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/sound.mat"
X=loadmat(file,struct_as_record=False)
sound = Sound(X['sound'].flatten())
sound.samplerate = 100*kHz
#sound = sound.atlevel(10*dB) # level in rms dB SPL
#X={}
#X['sound'] = sound.__array__()
#savemat('/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/sound.mat',X)

#sound = Sound(randn(1000))
#plot(sound)
#show()
#sound.samplerate = 100*kHz
cf = array([100*Hz,100*Hz,100*Hz,100*Hz])#erbspace(100*Hz, 1000*Hz, 50) # centre frequencies
cf = erbspace(100*Hz, 1000*Hz, 500) # centre frequencies

param_drnl = {}
#param_drnl['lp_nl_cutoff_m'] = 1.1
zilany_filter=ZILANY(sound, cf,32)
#zilany_filter=DRNL(sound, cf)
t1=time()
drnl = zilany_filter.process()
print time()-t1
#drnl =zilany_filter.rsigma
X={}
X['out_BM'] = drnl[:]
#X['out_BM'] = zilany_filter.rsigma
savemat('/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/out_BM.mat',X)

#figure()
subplot(211)
##print drnl[:]R
plot(drnl[:])

#imshow(flipud(drnl.T), aspect='auto')
subplot(212)
#print sound
plot(sound)
#imshow(flipud(dcgc.T), aspect='auto')
show()
