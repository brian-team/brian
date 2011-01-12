'''
Example of the possible interfacing of Brianhears with Brian.

Model of the spike generation at the auditory nerves. The sound is filtered by a bank of gammatone filters and is half wave rectified.
The output of this pheripheral model is fed to  leaky integrate and fire neurons, in particular the output of the filterbanks is the input current of the neurons.

'''
from brian import *  #needs brian
set_global_preferences(usenewbrianhears=True,
                       useweave=True,use_gpu = False)
from brian.hears import *

# Inner hair cell model 
cfmin, cfmax, cfN = 20*Hz, 20*kHz, 3000
cf = erbspace(cfmin, cfmax, cfN)
sound = Sound.whitenoise(100*ms) #generation of a white noise of 100ms long
gfb = Gammatone(sound, cf)
ihc = FunctionFilterbank(gfb, lambda x: 3*clip(x, 0, Inf)**(1.0/3.0))

# Leaky integrate-and-fire model with noise and refractoriness
eqs = '''
dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
I : 1
'''
G = FilterbankGroup(ihc, 'I', eqs, reset=0, threshold=1, refractory=5*ms)
# Run, and raster plot of the spikes
M = SpikeMonitor(G)
run(sound.duration)
raster_plot(M)
show()
