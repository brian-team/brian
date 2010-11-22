from brian import *

__all__ = ['get_samplerate', 'set_default_samplerate']

default_samplerate = 44.1*kHz

def get_samplerate(samplerate):
    if samplerate is None:
        return default_samplerate
    else:
        return samplerate
    
def set_default_samplerate(samplerate):
    '''
    Sets the default samplerate for Brian hears objects, by default 44.1 kHz.
    '''
    global default_samplerate
    default_samplerate = samplerate
