from brian import *
set_global_preferences(usenewbrianhears=True)
from brian.hears import *

x = Sound(randn(1000, 2), samplerate=44100*Hz)

gfb = GammatoneFilterbank(x, [1*kHz, 2*kHz])

ff = FunctionFilterbank(gfb, lambda input:clip(input, 0, Inf))

eqs = '''
dv/dt = (I-v)/(1*ms) : 1
I : 1
'''

G = FilterbankGroup(ff, 'I', eqs)
M = StateMonitor(G, 'v', record=True)

run(x.duration)

M.plot()
show()
