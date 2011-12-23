"""
A vocoder using Brian Hears
"""
from brian import *
from brian.hears import *
from scipy.signal import hilbert

S=vowel(vowel='u', pitch=150*Hz, duration=1*second, samplerate=None)
#S.play()
noise = whitenoise(1*second)
#sound.level = 50*dB

#(S+noise).play()

nbr_center_frequencies = 100
b1 = 1.019  #factor determining the time constant of the filters
center_frequencies = erbspace(30*Hz, 3000*Hz, nbr_center_frequencies)
bank1 = Gammatone(S, center_frequencies, b=b1)
bank2 = Gammatone(noise, center_frequencies, b=b1)

# Analytic signals
'''
Actually can't we directly get the analytic signals with the complex gammatone?
This would give an online calculation.
'''
H1=hilbert(bank1.process(),axis=0)
H2=hilbert(bank2.process(),axis=0)

envelope=abs(H1)
TFS=H2/abs(H2) # careful with division by zero

chimaera=sum(envelope*real(TFS),axis=1)

Sound(chimaera).play(normalise=True)

figure()
show()
