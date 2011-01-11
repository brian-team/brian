'''
Example of the use of the class LogGammachirp available in the library. It implements a filterbank of IIR gammachirp filters  as 
Unoki et al. 2001, "Improvement of an IIR asymmetric compensation gammachirp filter"
In this example, a white noise is filtered by a linear gammachirp filterbank and the resulting cochleogram is plotted. The different impulse responses are also plotted.
'''
from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *


dBlevel=50*dB  # dB level of the input sound in rms dB SPL
sound=whitenoise(100*ms,samplerate=44*kHz).ramp() #generation of a white noise
sound=sound.atlevel(dBlevel) #set the sound to a certain dB level

nbr_center_frequencies=50  #number of frequency channels in the filterbank

c1=-2.96 #linspace(-2,2,nbr_center_frequencies)    #glide slope
b1=1.81#linspace(1,1.5,nbr_center_frequencies)     #factor determining the time constant of the filters

cf=erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)  #center frequencies with a spacing following an ERB scale


gamma_chirp=LogGammachirp(sound,cf, c=c1,b=b1)     #instantiation of the filterbank 

gamma_chirp_mon=gamma_chirp.buffer_fetch(0, len(sound))  #processing. The results is a matrix.


figure()
imshow(flipud(gamma_chirp_mon.T),aspect='auto')    
show()    