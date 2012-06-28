import itertools

from brian import *
from brian.hears import *

def assert_sound_properties(snd, duration, samplerate, nchannels):
    '''
    Simple utility function to check whether all properties of a sound are
    as expected.
    '''    
    assert(isinstance(snd, Sound))
    
    assert(snd.duration == duration), '%s != %s' % (snd.duration, duration)
    assert(snd.samplerate == get_samplerate(samplerate)), '%s != %s' % (snd.samplerate, samplerate)
    assert(nchannels == nchannels), '%s != %s' % (snd.nchannels, nchannels)

def test_sound_construction():
    '''
    Tests various ways of instantiating sounds.
    '''
    
    # Test everything for default samplerate and a given one
    for samplerate in [None, 16 * kHz]:
        samplerate = get_samplerate(samplerate)
        # sound from array
        snd = Sound(array([-1, 0, 1]), samplerate=samplerate)
        
        assert_sound_properties(snd, 3 / samplerate, samplerate, 1)
    
        # stereo sound from array
        snd = Sound(array([[-1, 0, 1], [0, 1, 0]]).T, samplerate=samplerate)
        assert_sound_properties(snd, 3 / samplerate, samplerate, 2)
    
        # stereo sound from sequence
        snd = Sound([array([-1, 0, 1]), array([0, 1, 0])], samplerate=samplerate)
        assert_sound_properties(snd, 3 / samplerate, samplerate, 2)
        
        # sound from a function
        sound_func = lambda t: sin(2 * pi * 400 * Hz * t)
        snd = Sound(sound_func, samplerate=samplerate, duration=1*second)
        assert_sound_properties(snd, 1 * second, samplerate, 1)
        
    # Test everything for mono and stereo
    for samplerate, nchannels in itertools.product([None, 16 * kHz],
                                                   [1, 2]):
        samplerate = get_samplerate(samplerate)
        # tone
        snd = tone(400 * Hz, duration=1*second, samplerate=samplerate,
                   nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)
        
        # complex tone
        snd = harmoniccomplex(400 * Hz, duration=1*second,
                              samplerate=samplerate, nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)

        # whitenoise
        snd = whitenoise(duration=1*second, samplerate=samplerate,
                         nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)
        
        # pinknoise
        snd = pinknoise(duration=1*second, samplerate=samplerate,
                         nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)

        # brownnoise
        snd = brownnoise(duration=1*second, samplerate=samplerate,
                         nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)
        
        # powerlawnoise
        snd = powerlawnoise(1*second, 1, samplerate=samplerate,
                         nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)

        # currently fails, the duration is not correct
#        # irno
#        snd = irno(1 / (400 * Hz), 1, 3, duration=1*second,
#                   samplerate=samplerate, nchannels=nchannels)
#        assert_sound_properties(snd, 1 * second, samplerate, nchannels)
        
        # click
        snd = click(3 / samplerate, samplerate=samplerate, nchannels=nchannels)
        assert_sound_properties(snd, 3 / samplerate, samplerate, nchannels)
        
        # silence
        snd = silence(1 * second, samplerate=samplerate, nchannels=nchannels)
        assert_sound_properties(snd, 1 * second, samplerate, nchannels)
        
        # artificial vowel
        snd = vowel('a', duration=1 * second, samplerate=samplerate,
                    nchannels=nchannels)
        
if __name__ == '__main__':
    test_sound_construction()
    
    