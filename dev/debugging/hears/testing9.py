from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *
import time
start = time.time()
cfmin, cfmax, cfN = 20*Hz, 20*kHz, 3000
cf = erbspace(cfmin, cfmax, cfN)
sound = Sound.whitenoise(100*ms)
gfb = GammatoneFilterbank(sound, cf)
ihc = FunctionFilterbank(gfb, lambda x: 3*clip(x, 0, Inf)**(1.0/3.0))
eqs = '''
dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
I : 1
'''
G = FilterbankGroup(ihc, 'I', eqs, reset=0, threshold=1, refractory=5*ms)
M = SpikeMonitor(G)
run(sound.duration)
print time.time()-start
raster_plot(M)
show()
