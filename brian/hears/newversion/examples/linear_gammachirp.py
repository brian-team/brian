'''
Example of the use of the class LinearGammachirp available in the library. It implements a filterbank of FIR gammatone filters with linear frequency sweeps as 
described  in Wagner et al. 2009, "Auditory responses in the barn owl's nucleus laminaris to clicks: impulse response and signal analysis of neurophonic potential", J. Neurophysiol. 
In this example, a white noise is filtered by a gammachirp filterbank and the resulting cochleogram is plotted. The different impulse responses are also plotted.
'''
from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

dBlevel=50*dB  # dB level of the input sound in rms dB SPL
sound=whitenoise(100*ms,samplerate=44*kHz).ramp() #generation of a white noise
sound=sound.atlevel(dBlevel) #set the sound to a certain dB level

nbr_center_frequencies=10  #number of frequency channels in the filterbank
center_frequencies=erbspace(100*Hz, 1000*Hz, nbr_center_frequencies) #center frequencies with a spacing following an ERB scale

c=0.0 #linspace(-2,2,nbr_center_frequencies)   #glide slope
time_constant=linspace(3,0.3,nbr_center_frequencies)*ms

gamma_chirp=LinearGammachirp(sound, center_frequencies,time_constant,c) #instantiation of the filterbank 

gamma_chirp_mon=gamma_chirp.buffer_fetch(0, len(sound))  #processing. The results is a matrix.

figure()

imshow(gamma_chirp_mon.T,aspect='auto')    
figure()
plot(gamma_chirp.impulse_response.T)
show()    