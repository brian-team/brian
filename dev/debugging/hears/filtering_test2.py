from time import time
#http://amtoolbox.sourceforge.net/doc/filters/gammatone.php
from brian import *
set_global_preferences(useweave=True)
from scipy.io import loadmat,savemat

from brian.hears import *
#from zilany import *


simulation_duration = 100*ms
set_default_samplerate(50*kHz)
sound = whitenoise(simulation_duration)
#file="/home/bertrand/Data/MatlabProg/brian_hears/Carney/sound.mat"
#X=loadmat(file,struct_as_record=False)
#sound = Sound(X['sound'].flatten())
sound.samplerate = 50*kHz
sound = sound.atlevel(120*dB) # level in rms dB SPL
X={}
X['sound'] = sound.__array__()
savemat('/home/bertrand/Data/MatlabProg/brian_hears/Carney/sound.mat',X)

#sound = Sound(randn(1000))
#plot(sound)
#show()
#sound.samplerate = 100*kHz
cf = array([1000*Hz])#erbspace(100*Hz, 1000*Hz, 50) # centre frequencies
#cf = erbspace(100*Hz, 1000*Hz, 500) # centre frequencies

param_drnl = {}
#param_drnl['lp_nl_cutoff_m'] = 1.1
zilany_filter=TAN(sound, cf,1)
#zilany_filter=DRNL(sound, cf)
t1=time()
drnl = zilany_filter.process()
print time()-t1
print drnl.shape
#drnl =zilany_filter.control_cont 
#drnl =zilany_filter.signal_cont 
X={}
X['out_BM'] = drnl[:]
#X['out_BM'] = zilany_filter.param
savemat('/home/bertrand/Data/MatlabProg/brian_hears/Carney/out_BM.mat',X)

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
