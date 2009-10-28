from brian import *
from numpy import *
import numpy
import wave
import array as pyarray
import time
try:
    import pygame
    have_pygame = True
except ImportError:
    have_pygame = False
try:
    from scikits.samplerate import resample
    have_scikits_samplerate = True
except ImportError:
    have_scikits_samplerate = False

__all__ = ['Sound', 'play_stereo_sound',
           'whitenoise', 'tone', 'click', 'silent', 'sequence','mix_sounds']

class Sound(numpy.ndarray):
    duration = property(fget=lambda self:len(self)/self.rate)
    times = property(fget=lambda self:arange(len(self), dtype=float)/self.rate)
    
    @check_units(rate=Hz, duration=second)
    def __new__(cls, data, rate=None, duration=None):
        if isinstance(data, numpy.ndarray):
            if rate is None:
                raise ValueError('Must specify rate to initialise Sound with array.')
            if duration is not None:
                raise ValueError('Cannot specify duration when initialising Sound with array.')
            x = array(data, dtype=float)
        elif isinstance(data, str):
            if duration is not None:
                raise ValueError('Cannot specify duration when initialising Sound from file.')
            rate, x = get_wav(data, rate)
        elif callable(data):
            if rate is None:
                raise ValueError('Must specify rate to initialise Sound with function.')
            if duration is None:
                raise ValueError('Must specify duration to initialise Sound with function.')
            L = int(duration*rate)
            t = arange(L, dtype=float)/rate
            x = data(t)
        else:
            raise TypeError('Cannot initialise Sound with data of class '+str(data.__class__))
        x = x.view(cls)
        x.rate = rate
        return x

    def __array_wrap__(self, obj, context=None):
        handled = False
        x = numpy.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'rate') and hasattr(self, 'rate'):
            x.rate = self.rate
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x
    
    def __add__(self, other):
        if isinstance(other, Sound):
            if int(other.rate)>int(self.rate):
                self = self.resample(other.rate)
            elif int(other.rate)<int(self.rate):
                other = other.resample(self.rate)

            if len(self)>len(other):
                other = other.extend_length(len(self))
            elif len(self)<len(other):
                self = self.extend_length(len(other))
                
            return Sound(numpy.ndarray.__add__(self, other), rate=self.rate)
        else:
            x = numpy.ndarray.__add__(self, other)
            return Sound(x, self.rate)
    __radd__ = __add__
    
    @check_units(duration=second)
    def extend(self, duration):
        '''
        Returns the Sound with length extended (or contracted) to have the given duration.
        '''
        L = int(duration*self.rate)
        return self.extend_length(L)
    
    def extend_length(self, L):
        '''
        Returns the Sound with length extended (or contracted) to have L samples.
        '''
        if L==len(self):
            return self
        elif L<len(self):
            return Sound(self[:L], rate=self.rate)
        else:
            return Sound(concatenate((self,zeros(L-len(self)))), rate=self.rate)
    
    @check_units(onset=second)
    def shift(self, onset):
        '''
        Returns the sound played at time onset.
        '''
        _, x = mix_sounds(self.rate, (self, onset))
        return Sound(x, rate=self.rate)
    
    def repeat(self, n):
        '''
        Repeats the sound n times
        '''
        x = hstack((self,)*n)
        return Sound(x, rate=self.rate)
    
    @check_units(rate=Hz)
    def resample(self, rate, resample_type='sinc_best'):
        '''
        Returns a resampled version of the sound.
        '''
        rate, x = resample_sound(self, self.rate, rate, resample_type=resample_type)
        return Sound(x, rate=rate)
    
    def copy_from(self, other):
        '''
        Copies values from the given sound (resampled if necessary).
        '''
        if not isinstance(other, Sound):
            raise TypeError('Must copy from a Sound object.')
        if int(other.rate)!=int(self.rate):
            other = other.resample(self.rate)
        self[:min(len(other),len(self))] = other[:min(len(other),len(self))]
    
    def play(self, normalise=False, sleep=False):
        '''
        Plays the sound (normalised to avoid clipping if required).
        '''
        if normalise:
            play_sound(self.rate, self/amax(abs(self)))
        else:
            play_sound(self.rate, self)
        if sleep:
            time.sleep(self.duration)
        
    def spectrogram(self, frequency_range=None, log_spectrogram=True, **kwds):
        '''
        Plots a spectrogram of the sound
        
        If frequency_range=None it shows the full spectrogram, otherwise
        frequency_range=(minfreq, maxfreq).
        
        If log_spectrogram=True it shows log power, otherwise not.
        
        kwds are passed to pylab's specgram command.
        
        Returns the values returned by pylab's specgram, namely pxx, freqs, bins, im
        where pxx is a 2D array of powers, freqs is the corresponding frequences, bins
        are the time bins, and im is the image axis.
        '''
        pxx, freqs, bins, im = specgram(self, Fs=self.rate, **kwds)
        if frequency_range is not None:
            I = logical_and(frequency_range[0]<=freqs,freqs<=frequency_range[1])
            I2 = where(I)[0]
            I2 = [max(min(I2)-1,0), min(max(I2)+1,len(freqs)-1)]
            Z = pxx[I2[0]:I2[-1],:]
        else:
            Z = pxx
        if log_spectrogram:
            Z[Z<1e-20] = 1e-20 # no zeros because we take logs
            Z = 10*log10(Z)
        Z = flipud(Z)
        if frequency_range is not None:
            imshow(Z, extent=(0, amax(bins), freqs[I2[0]], freqs[I2[-1]]), aspect='auto')
        else:
            imshow(Z, extent=(0, amax(bins), freqs[0], freqs[-1]), aspect='auto')
        xlabel('Time (s)')
        ylabel('Frequency (Hz)')
        return (pxx, freqs, bins, im)
    
    def intensity(self):
        '''
        Returns intensity in dB SPL assuming array is in Pascals
        '''
        return 20.0*log10(sqrt(mean(asarray(self)**2))/2e-5)
    
    def atintensity(self, db):
        '''
        Returns sound in Pascals at various intensities (in RMS dB SPL)
        '''
        return self.amplified(db-self.intensity())
        
    def amplified(self, db):
        '''
        Returns sound amplified by a given amount in dB pressure.
        '''
        return self*10.0**(db/20.0)
    
    def ramp(self, when='both', duration=10*ms, func=None, inplace=False):
        '''
        Adds a ramp on/off to the sound
        
        ``when='on'``
            Can take values 'on', 'off' or 'both'
        ``duration=10*ms``
            The time over which the ramping happens
        ``func``
            A ramping function, if not specified uses ``sin(pi*t/2)**2``
        ``inplace``
            Whether to apply ramping to current sound or return a new array.
        '''
        when = when.lower().strip()
        if func is None: func = lambda t:sin(pi*t/2)**2
        sz = int(duration*self.rate)
        multiplier = func(linspace(0.0, 1.0, sz))
        if inplace:
            target = self
        else:
            target = Sound(copy(self), self.rate)
        if when=='on' or when=='both':
            target[:sz] *= multiplier
        if when=='off' or when=='both':
            target[-sz:] *= multiplier[::-1]
        return target
    
    @staticmethod
    def tone(freq, duration, rate=44.1*kHz):
        '''
        Returns a pure tone at the given frequency for the given duration
        '''
        rate, x = make_puretone(freq, duration, rate)
        return Sound(x, rate)

    @staticmethod
    def whitenoise(duration, rate=44.1*kHz):
        '''
        Returns a white noise for the given duration.
        '''
        rate, x = make_whitenoise(duration, rate)
        return Sound(x, rate)
    
    @staticmethod
    def click(duration, amplitude=1, rate=44.1*kHz):
        '''
        Returns a click with given parameters
        '''
        rate, x = make_click(duration, amplitude, rate)
        return Sound(x, rate)
    
    @staticmethod
    def silent(duration, rate=44.1*kHz):
        '''
        Returns a silent, zero sound for the given duration.
        '''
        x = numpy.zeros(int(duration*rate))
        return Sound(x, rate)
    
    @staticmethod
    def sequence(sounds, rate):
        '''
        Returns the sequence of sounds in the list sounds joined together
        '''
        sounds = tuple(s.resample(rate) for s in sounds)
        x = hstack(sounds)
        return Sound(x, rate)
    
    def save(self, filename, normalise=False, sampwidth=1):
        '''
        Save the sound as a WAV file.
        
        If the normalise keyword is set to True, the amplitude of the sound will be
        normalised to 1. The sampwidth keyword can be 1 or 2 to save the data as
        8 or 16 bit samples.
        '''
        if sampwidth!=1 and sampwidth!=2:
            raise ValueError('Sample width must be 1 or 2 bytes.')
        typecode = {2:'h', 1:'B'}[sampwidth]
        scale = {2:2**14, 1:2**7}[sampwidth]
        meanval = {2:0, 1:1}[sampwidth]
        dtype = {2:int16, 1:uint8}[sampwidth]
        w = wave.open(filename, 'wb')
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(int(self.rate))
        w.setnframes(len(self))
        x = asarray(self)
        if normalise:
            x /= amax(x)
        x = (x+meanval)*scale
        data = array(x, dtype=dtype)
        data = pyarray.array(typecode, data)
        w.writeframes(data)
        w.close()

    def __reduce__(self):
        return (_load_Sound_from_pickle, (asarray(self), float(self.rate)))

def _load_Sound_from_pickle(arr, rate):
    return Sound(arr, rate=rate*Hz)
        

def get_wav(filename, forced_rate=None):
    '''
    get_wav(filename) returns (rate, data)
    
    Data is a numpy float array with values between -1 and 1.
    Currently, 8 and 16 bit mono sounds are handled correctly.
    Specify forced_rate to resample to a given rate (in a very crude
    way).
    '''
    f = wave.open(filename,'r')
    sampwidth = f.getsampwidth()
    typecode = {2:'h', 1:'B'}[sampwidth]
    scale = {2:2**14, 1:2**7}[sampwidth]
    meanval = {2:0, 1:1}[sampwidth]
    rate = f.getframerate()
    x = f.readframes(f.getnframes())
    x = pyarray.array(typecode, x)
    x = array(x, dtype=float)
    x = x/scale
    x = x-meanval
    f.close()
    rate = rate*Hz
    if forced_rate is not None:
        rate, x = resample_sound(x, rate, forced_rate)
    return (rate, x)

@check_units(duration=second, sound_rate=Hz)
def make_whitenoise(duration, sound_rate = 44100*Hz):
    '''
    Make a white noise signal with unit variance, returns (rate, x)
    '''
    sound_x = randn(int(sound_rate*duration))
    return (sound_rate, sound_x)

@check_units(tonefreq=Hz, duration=second, sound_rate=Hz)
def make_puretone(tonefreq, duration, sound_rate = 44100*Hz):
    '''
    Make a pure tone signal, returns (rate, x)
    '''
    sound_x = sin(2.0*pi*tonefreq*arange(0*ms, duration, 1/sound_rate))
    return (sound_rate, sound_x)

@check_units(duration=second, sound_rate=Hz)
def make_click(duration, amplitude=1, sound_rate = 44100*Hz):
    '''
    Make a click signal, returns (rate, x)
    '''
    sound_x = amplitude*ones(int(duration*sound_rate))
    return (sound_rate, sound_x)

@check_units(oldrate=Hz, newrate=Hz)
def resample_sound(x, oldrate, newrate, resample_type='sinc_best'):
    if not have_scikits_samplerate:
        raise ImportError('Need scikits.samplerate package for resampling')
    y = array(resample(x, float(newrate/oldrate), resample_type), dtype=float64)
    return (newrate, y)

@check_units(rate=Hz)
def mix_sounds(rate, *sounds):
    '''
    Mixes sound sources
    
    All the sounds must have the same sampling rate, call as::
    
        rate, x = mix_sounds(rate,
                    (sound1, offset1),
                    (sound2, offset2),
                    ...)
    '''
    duration = max(len(x)+offset*rate for x, offset in sounds)
    y = zeros(int(duration)+1)
    for x, offset in sounds:
        o = int(offset*rate)
        if o<0:
            x = x[-o:]
            o = 0
        y[o:o+len(x)] += x
    return (rate, y)

@check_units(rate=Hz)
def play_sound(rate, x):
    if have_pygame:
        pygame.mixer.init(int(rate), -16, 1)
        y = array((2**15-1)*clip(x,-1,1), dtype=int16)
        s = pygame.sndarray.make_sound(y)
        s.play()
    else:
        warnings.warn('Cannot play sound, no pygame module.')

def play_stereo_sound(sound_l, sound_r, sleep=False):
    pygame.mixer.quit()
    pygame.mixer.init(int(sound_l.rate), -16, 2)
    a = max(amax(abs(sound_l)), amax(abs(sound_r)))
    xl = array(2**15*clip(sound_l/a,-1,1), dtype=int16)
    xr = array(2**15*clip(sound_r/a,-1,1), dtype=int16)
    x = vstack((xl,xr)).T.copy()
    x = pygame.sndarray.make_sound(x)
    x.play()
    if sleep:
        time.sleep(sound_l.duration)

whitenoise = Sound.whitenoise
tone = Sound.tone
click = Sound.click
silent = Sound.silent
sequence = Sound.sequence

if __name__=='__main__':

    import time
    x = tone(500*Hz, 1*second).atintensity(rmsdb=80.)
    
    print amax(x)
    print 20*log10(sqrt(mean(x**2))/2e-5)
    print x.intensity()
    
    #x.save('tone.wav', sampwidth=2)
    #x.play()
    #time.sleep(x.duration)

#    x = Sound('../sounds/downloads/terminator.wav')
#    y = Sound('../sounds/ens/Bassoon.A3.wav')
#    z = x + y + 10*y.shift(1*second) + Sound.tone(500*Hz, 1*second) + 0.1*Sound.whitenoise(1.5*second).shift(1.5*second)
#    x = x.shift(-1*second)
#    x += 0.1*Sound.whitenoise(x.duration, x.rate)
#    x = x.repeat(10)
#    x.play()
#    import time
#    time.sleep(x.duration)

#    from __builtin__ import sum
#    z = sum(Sound.tone(20*Hz+10*kHz*rand(), 1*second*rand()).shift(10*second*rand()) for _ in range(10))

#    z = Sound(lambda t:sin(2*pi*200*t), duration=1*second, rate=44.1*kHz)
    
#    z.play(True)
#    subplot(311)
#    plot(z.times, z)
#    axis(xmax=float(z.duration))
#    subplot(312)
#    z.spectrogram()
#    subplot(313)
#    z.spectrogram(frequency_range=(5*kHz,10*kHz))
#    show()