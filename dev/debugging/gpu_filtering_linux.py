from brian import *
from brian.hears import*
from brian.hears import filtering
#filtering.use_gpu = False 

samplerate = 44.1*kHz
defaultclock.dt = 1/samplerate

cfmin, cfmax, cfN = 100*Hz, 2*kHz, 5
cf = erbspace(cfmin, cfmax, cfN)

sound = Sound(array([1]), rate=samplerate).extend(50*ms)

fb = GammatoneFilterbank(samplerate, cf)

G = FilterbankGroup(fb, sound)

M = StateMonitor(G, 'output', record=True)

run(sound.duration)

M.plot()
show()
