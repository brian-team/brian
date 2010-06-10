from brian import *
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False 
import scipy.signal as signal

'''
Compute the frequency response of a gammachirp, of its  asymmetric compensation filter, and  of its gammatone filter
at 2000Hz

The script reproduces figure 1 from Irino and Unoki, AN ANALYSIS/SYNTHESIS AUDITORY FILTERBANK BASED ON AN IIR GAMMACHIRP FILTER
but in the paper the figure is an FIR implementation (no error). The one presented here is the approximated IIR filter


'''

samplerate=44*kHz

center_frequencies=array([2000*Hz])

c=-2  #c determines the rate of the frequency modulation or the chirp rate (c=0 is a gammtone filter)

gammachirp =GammachirpFilterbankIIR(samplerate,center_frequencies,c=c )


gammatone_filt_b=gammachirp.gammatone_filt_b
gammatone_filt_a=gammachirp.gammatone_filt_a

asymmetric_filt_b=gammachirp.asymmetric_filt_b
asymmetric_filt_a=gammachirp.asymmetric_filt_a
 


gammachirp_filt_b=gammachirp.filt_b
gammachirp_filt_a=gammachirp.filt_a


## the indivisual filters are cascade, so the frequency response is multiply
hgammatone=1
for iorder in range((gammatone_filt_b.shape[2])):
   w,htemp = signal.freqz(gammatone_filt_b[0,:,iorder],gammatone_filt_a[0,:,iorder])
   hgammatone=hgammatone*htemp
   
hasym=1
for iorder in range((asymmetric_filt_b.shape[2])):
   w,htemp = signal.freqz(asymmetric_filt_b[0,:,iorder],asymmetric_filt_a[0,:,iorder])
   hasym=hasym*htemp
   
hgammachirp=1
for iorder in range((gammachirp_filt_b.shape[2])):
   w,htemp = signal.freqz(gammachirp_filt_b[0,:,iorder],gammachirp_filt_a[0,:,iorder])
   hgammachirp=hgammachirp*htemp

hgammatone_dB = 20 * log10 (abs(hgammatone))
hasym_dB = 20 * log10 (abs(hasym))
hgammachirp_dB = 20 * log10 (abs(hgammachirp))

#hgammatone_dB = abs(hgammatone)
#hasym_dB = abs(hasym)
#htotal_dB = abs(htotal)

w=w/2/pi*samplerate
ind=nonzero(w<=4000)

subplot(311)
plot(w[ind],hgammachirp_dB[ind])
ylim(-60, 0)
ylabel('Magnitude (db)')
figtext(0.7,0.8,'Gammachirp')



subplot(312)
plot(w[ind],hgammatone_dB[ind])
ylim(-60, 0)
ylabel('Magnitude (db)')
figtext(0.7,0.5,'Gammatone')

subplot(313)
plot(w[ind],hasym_dB[ind])
ylim(-35, 30)
ylabel('Magnitude (db)')
xlabel(r'Frequeny [Hz]')
figtext(0.55,0.2,'Asymmetric compensation filter')
show()