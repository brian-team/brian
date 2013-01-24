from numpy.testing.utils import assert_raises, assert_equal

from brian import *
from brian.tests import repeat_with_global_opts
from brian.hears import *

def assert_sound_properties(snd, duration, samplerate, nchannels):
    '''
    Simple utility function to check whether all properties of a sound are
    as expected.
    '''    
    assert(isinstance(snd, Sound))
    
    assert(snd.duration == duration), '%s != %s' % (snd.duration, duration)
    assert(snd.samplerate == get_samplerate(samplerate)), '%s != %s' % (snd.samplerate, samplerate)
    assert(snd.nchannels == nchannels), '%s != %s' % (snd.nchannels, nchannels)

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
    # 
    # Note (do not use itertools.product here as this does not exist in 
    # python 2.5)
    for samplerate, nchannels in [(None, 1), (None, 2), (16*kHz, 1), (16*kHz, 2)]:
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

def test_sound_access():
    '''
    Test accessing attributes and parts of a sound in various ways.
    '''
    
    for nchannels in [1, 2]:
        if nchannels == 1:
            snd = tone(400*Hz, duration=1*second)
        else:
            snd = Sound([tone(400*Hz, duration=1*second),
                         tone(600*Hz, duration=1*second)])
        assert(snd.nsamples == int(snd.duration * snd.samplerate))
        assert(len(snd.times) == snd.nsamples)
        
        if nchannels == 1:
            assert(isinstance(snd.level, dB_type))
        
        if nchannels == 2:
            assert(np.all(snd.left == snd.channel(0)))
            assert(np.all(snd[:, 0] == snd.channel(0)))
            assert(np.all(snd.right == snd.channel(1)))
            assert(np.all(snd[:, 1] == snd.channel(1)))
        
        snd.level = 60.0 * dB
        
        # Changing level should not change other properties
        assert_sound_properties(snd.atlevel(70*dB), snd.duration,
                                snd.samplerate, snd.nchannels)
        assert_sound_properties(snd.atmaxlevel(80*dB), snd.duration,
                                snd.samplerate, snd.nchannels)
        
        assert_sound_properties(snd.ramped(), snd.duration, snd.samplerate,
                               snd.nchannels)
        
        assert(len(snd[:100]) == 100)
        assert(len(snd[:snd.duration]) == len(snd))
        assert_raises(DimensionMismatchError, lambda :snd[:3*Hz])
        
        snd[:100, :] = 0
        assert(np.all(snd[:100, :] == 0))
        
        # repeat
        snd_repeated = snd.repeat(3)
        assert_sound_properties(snd_repeated, snd.duration * 3,
                                snd.samplerate, snd.nchannels)
        snd_sequence = sequence([snd, snd])
        assert_sound_properties(snd_sequence, snd.duration * 2,
                                snd.samplerate, snd.nchannels)

def test_shifting():
    # one channel
    snd = tone(1000*Hz, duration=100*ms, samplerate=10*kHz)
    shifted = snd.shifted(10*ms)
    assert shifted.duration == 110*ms
    assert_equal(shifted[:10*ms], silence(duration=10*ms, samplerate=10*kHz))
    assert_equal(shifted[10*ms:], snd)
    
    #fractional shift
    snd = tone(1000*Hz, duration=100*ms, samplerate=10*kHz)
    shifted = snd.shifted(10.0*ms, fractional=True)    
    # actually that leads to a sound with a total length of 109.9ms, 
    # I think for multiples of the sample rate, using fractional or not should
    # not make a difference for the length...   
    assert_equal(shifted[:10*ms], silence(duration=10*ms, samplerate=10*kHz))
    
    # several channels
    snd = Sound([tone(1000*Hz, duration=100*ms, samplerate=10*kHz),
                 tone(2000*Hz, duration=100*ms, samplerate=10*kHz)],
                samplerate=10*kHz)
    shifted = snd.shifted(10*ms)
    assert shifted.nchannels == snd.nchannels
    assert shifted.duration == 110*ms
    for channel in xrange(shifted.nchannels):
        assert_equal(shifted[:10*ms, channel],
                     silence(duration=10*ms, samplerate=10*kHz))
        assert_equal(shifted[10*ms:, channel], snd[:, channel])        
    
    #fractional shift
    shifted = snd.shifted(10.0*ms, fractional=True)
    assert shifted.nchannels == snd.nchannels
    for channel in xrange(shifted.nchannels):
        assert_equal(shifted[:10*ms, channel],
                     silence(duration=10*ms, samplerate=10*kHz))
    
@repeat_with_global_opts([{'useweave': False},
                          {'useweave': True}])
def test_linear_filtering():
    ''' Test that linear filtering does not change the source '''
    original_sound = Sound(linspace(-1, 1, 1000))
    sound = original_sound.copy()
    filtered = Gammatone(sound, 1000*Hz).process()
    assert_equal(np.asarray(original_sound), np.asarray(sound))


@repeat_with_global_opts([{'useweave': False},
                          {'useweave': True}])
def test_multichannel_processing():
    '''Make sure that processing multiple channels gives the same results as
       processing single channels.
    '''
    set_default_samplerate(50*kHz)
    freqs = np.array([100, 1000, 5000])
    sounds = Sound([harmoniccomplex(float(freq)*Hz,
                                    duration=5*ms) for freq in freqs])
    # test some filters that only need a single 'CF' argument in addition
    # to the source 
    filters = [Gammatone, DRNL, TanCarney]
    for one_filter in filters:
        print one_filter.__name__
        # process all channels in parallel
        all_channels = one_filter(sounds, freqs).process()
        all_channels_array = np.asarray(all_channels)
        # process each channel separately 
        for idx, freq in enumerate(freqs):
            one_channel = one_filter(sounds.channel(idx), freq).process()
            assert_equal(np.asarray(one_channel).flatten(),
                         all_channels_array[:, idx])


@repeat_with_global_opts([{'useweave': False},
                          {'useweave': True}])
def test_middleear():
    '''Make sure that using multiple gain values in the middle ear model do the
       expected thing.
    '''
    gains = np.arange(1., 5.)
    snd = harmoniccomplex(1000*Hz, duration=5*ms)
    # use multiple gains for the same sound
    all_gains = MiddleEar(snd, gain=gains).process()
    # same but with RestructureFilterbank
    all_gains2 = MiddleEar(RestructureFilterbank(snd, len(gains)),
                           gain=gains).process()
    assert_equal(np.asarray(all_gains), np.asarray(all_gains2))
    
    # use single gain values
    for idx, gain in enumerate(gains):
        one_gain = MiddleEar(snd, gain=gain).process()
        assert_equal(np.asarray(all_gains)[:, idx].flatten(),
                     np.asarray(one_gain).flatten())


if __name__ == '__main__':
    test_sound_construction()
    test_sound_access()
    test_shifting()
    test_linear_filtering()
    test_multichannel_processing()
    test_middleear()
