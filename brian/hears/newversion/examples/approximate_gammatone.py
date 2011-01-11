'''
Example of the use of the class ApproximateGammatone available in the library. It implements a filterbank of approximate gammatone filters as 
described  in Hohmann, V., 2002, "Frequency analysis and synthesis using a Gammatone filterbank", Acta Acustica United with Acustica. 
In this example, a white noise is filtered by a gammatone filterbank and the resulting cochleogram is plotted.
'''
from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *

dBlevel=50*dB  # dB level of the input sound in rms dB SPL
sound=whitenoise(100*ms,samplerate=44*kHz).ramp() #generation of a white noise
sound=sound.atlevel(dBlevel) #set the sound to a certain dB level

nbr_center_frequencies=50  #number of frequency channels in the filterbank
center_frequencies=erbspace(100*Hz, 1000*Hz, nbr_center_frequencies) #center frequencies with a spacing following an ERB scale
bw=10**(0.037+0.785*log10(center_frequencies))   #bandwidth of the filters (different in each channel)

gammatone=ApproximateGammatone(sound, center_frequencies, bw,order=3) #instantiation of the filterbank 

gt_mon=gammatone.buffer_fetch(0, len(sound))  #processing. The results is a matrix.

figure()
imshow(flipud(gt_mon.T),aspect='auto')    
show()




    