'''
Example of the use of the cochlear models (:class:`~brian.hears.DRNL` and
:class:`~brian.hears.DCGC`) available in the library.
'''
from brian import *
set_global_preferences(useweave=True)
from scipy.io import loadmat,savemat

from brian.hears import *
#from zilany import *

def test():
    simulation_duration = 1*ms
    set_default_samplerate(100*kHz)
    #sound = whitenoise(simulation_duration)
    file="/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/sound.mat"
    X=loadmat(file,struct_as_record=False)
    sound = Sound(X['sound'].flatten())
    sound.samplerate = 100*kHz
    sound = sound.atlevel(10*dB) # level in rms dB SPL
    X={}
    X['sound'] = sound.__array__()
    savemat('/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/sound.mat',X)
    
    sound = Sound(randn(100))
    #plot(sound)
    #show()
    #sound.samplerate = 100*kHz
    cf = array([1000*Hz])#erbspace(100*Hz, 1000*Hz, 50) # centre frequencies
    #cf = erbspace(100*Hz, 1000*Hz, 50) # centre frequencies
    
    param_drnl = {}
    #param_drnl['lp_nl_cutoff_m'] = 1.1
    zilany_filter=ZILANY(sound, cf,1)
    #zilany_filter=DRNL(sound, cf)
    drnl = zilany_filter.process()
    
    X={}
    X['out_BM'] = drnl[:]
    savemat('/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/out_BM.mat',X)
    
    #figure()
    subplot(211)
    ##print drnl[:]R
    plot(drnl[:])
    
    #imshow(flipud(drnl.T), aspect='auto')
    subplot(212)
    #print sound
    plot(sound)
    #imshow(flipud(dcgc.T), aspect='auto')
    show()
import cProfile
cProfile.run('test()')