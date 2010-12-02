from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from time import time

dBlevel=60  # dB level in rms dB SPL
sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
samplerate=sound.samplerate
sound=sound.atintensity(dBlevel)
sound.samplerate=samplerate

print 'fs=',sound.samplerate,'duration=',len(sound)/sound.samplerate

simulation_duration=len(sound)/sound.samplerate


sound=sound*0.00014  #conversion to stape velocity
sound.samplerate=samplerate
data=dict()
data['input']=sound
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/stimulus.mat',data)



simulation_duration=len(sound)/samplerate

nbr_center_frequencies=50
center_frequencies=log_space(100*Hz, 1000*Hz, nbr_center_frequencies)


##### middle ear processing ####
#f_cut_off1=25*kHz/ ( ( 10**(0.1*10)-1)**(1.0/(2.0*2))) #Find the butterworth natural frequency W0 from the lower and higher cutt off 
#f_cut_off2=30*kHz/ ( ( 10**(0.1*10)-1)**(1.0/(2.0*2)))
#lp_l1=ButterworthFilterbank(samplerate, 1, 2,f_cut_off1, btype='low')
#lp_l2=ButterworthFilterbank(samplerate, 1, 3,f_cut_off2, btype='low')
##middle_ear=FilterbankChain([lp_l1,lp_l2])
#middle_ear=DoNothingFilterbank(samplerate, 1)

##### conversion from pressure (Pascal) in stapes velocity (in m/s)
#stape_scale=0.00014
#stapes_fun=lambda x:x*stape_scale
#stapes_conv=FunctionFilterbank(samplerate, 1, stapes_fun)

#### Linear Pathway ####



#bandpass filter (second order  gammatone filter)
center_frequencies_linear=10**(-0.067+1.016*log10(center_frequencies))
bandwidth_linear=10**(0.037+0.785*log10(center_frequencies))
order_linear=3
gammatone=ApproximateGammatoneFilterbank(sound, center_frequencies_linear, order_linear, bandwidth_linear)

#linear gain
g=10**(4.2-0.48*log10(center_frequencies))
#g=10**(4.2-0.48*log10(tile(center_frequencies,(len(sound),1))))
func_gain=lambda x:g*x
gain= FunctionFilterbank(gammatone,func_gain)

#low pass filter(cascade of 4 second order lowpass butterworth filters)
cutoff_frequencies_linear=center_frequencies_linear
order_lowpass_linear=2
lp_l=LowPassFilterbank(gain, nbr_center_frequencies,cutoff_frequencies_linear)
lowpass_linear=CascadeFilterbank(gain,lp_l,4)


#### Nonlinear Pathway ####

#bandpass filter (third order gammatone filters)
center_frequencies_nonlinear=center_frequencies#10**(-0.05252+1.0165*log10(center_frequencies))
bandwidth_nonlinear=10**(-0.031+0.774*log10(center_frequencies))
order_nonlinear=3
bandpass_nonlinear1=ApproximateGammatoneFilterbank(sound, center_frequencies_nonlinear, order_nonlinear, bandwidth_nonlinear)

#compression (linear at low level, compress at high level)
a=10**(1.402+0.819*log10(center_frequencies))  #linear gain
b=10**(1.619-0.818*log10(center_frequencies))  
v=.2 #compression exponent
func_compression=lambda x:sign(x)*minimum(a*abs(x),b*abs(x)**v)
compression=FunctionFilterbank(bandpass_nonlinear1,  func_compression)

#bandpass filter (third order gammatone filters)
bandpass_nonlinear2=ApproximateGammatoneFilterbank(compression, center_frequencies_nonlinear, order_nonlinear, bandwidth_nonlinear)

#low pass filter
cutoff_frequencies_nonlinear=center_frequencies_nonlinear
order_lowpass_nonlinear=2
lp_nl=LowPassFilterbank(bandpass_nonlinear2, nbr_center_frequencies,cutoff_frequencies_nonlinear)

#lp_nl=ButterworthFilterbank(samplerate, nbr_center_frequencies, order_lowpass_nonlinear, cutoff_frequencies_nonlinear, btype='low')
lowpass_nonlinear=CascadeFilterbank(bandpass_nonlinear2,lp_nl,3)

#### adding the two pathways together ####
#dnrl_filter = RestructureFilterbank((lowpass_linear, lowpass_nonlinear))



dnrl_filter=lowpass_linear+lowpass_nonlinear


dnrl_filter.buffer_init()
t1=time()
dnrl=dnrl_filter.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'

data=dict()
data['out']=dnrl.T
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/drnl_BH.mat',data)

figure()
imshow(flipud(dnrl.T),aspect='auto')    
show()


