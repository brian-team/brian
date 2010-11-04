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
except (ImportError, ValueError):
    have_scikits_samplerate = False
from bufferable import Bufferable

__all__ = ['Sound',
           'whitenoise', 'tone', 'click', 'silent', 'sequence',
           ]

class BaseSound(Bufferable):
    '''
    Base class for Sound and OnlineSound
    '''
    pass

class Sound(BaseSound, numpy.ndarray):
    '''
    TODO: documentation for Sound
    '''
    duration = property(fget=lambda self:len(self) / self.samplerate)
    times = property(fget=lambda self:arange(len(self), dtype=float) / self.samplerate)
    nchannels = property(fget=lambda self:self.shape[1])
    left = property(fget=lambda self:self.channel(0))
    right = property(fget=lambda self:self.channel(1))

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
            x = Sound.load(data, rate=rate)
            rate = x.samplerate
        elif callable(data):
            if rate is None:
                raise ValueError('Must specify rate to initialise Sound with function.')
            if duration is None:
                raise ValueError('Must specify duration to initialise Sound with function.')
            L = int(duration * rate)
            t = arange(L, dtype=float) / rate
            x = data(t)
        elif isinstance(data, (list, tuple)):
            channels = tuple(Sound(c, rate=rate, duration=duration) for c in data)
            x = hstack(channels)
            rate = channels[0].samplerate
        else:
            raise TypeError('Cannot initialise Sound with data of class ' + str(data.__class__))
        if len(x.shape)==1:
            x.shape = (len(x), 1)
        x = x.view(cls)
        x.samplerate = rate
        x.buffer_init()
        return x

    def __array_wrap__(self, obj, context=None):
        handled = False
        x = numpy.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'rate') and hasattr(self, 'rate'):
            x.samplerate = self.samplerate
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x
    
    def buffer_init(self):
        pass
        
    def buffer_fetch(self, start, end):
        if start<0:
            raise IndexError('Can only use positive indices in buffer.')
        samples = end-start
        X = asarray(self)[start:end, :]
        if X.shape[0]<samples:
            X = vstack((X, zeros((samples-X.shape[0], X.shape[1]))))
        return X

    def channel(self, n):
        return Sound(self[:, n], self.samplerate)

    def __add__(self, other):
        if isinstance(other, Sound):
            if int(other.samplerate) > int(self.samplerate):
                self = self.resample(other.samplerate)
            elif int(other.samplerate) < int(self.samplerate):
                other = other.resample(self.samplerate)

            if len(self) > len(other):
                other = other.extend_length(len(self))
            elif len(self) < len(other):
                self = self.extend_length(len(other))

            return Sound(numpy.ndarray.__add__(self, other), rate=self.samplerate)
        else:
            x = numpy.ndarray.__add__(self, other)
            return Sound(x, self.samplerate)
    __radd__ = __add__

    @check_units(duration=second)
    def extend(self, duration):
        '''
        Returns the Sound with length extended (or contracted) to have the given duration.
        '''
        L = int(duration * self.samplerate)
        return self.extend_length(L)

    def extend_length(self, L):
        '''
        Returns the Sound with length extended (or contracted) to have L samples.
        '''
        if L == len(self):
            return self
        elif L < len(self):
            return Sound(self[:L, :], rate=self.samplerate)
        else:
            padding = zeros((L - len(self), self.nchannels))
            return Sound(concatenate((self, padding)), rate=self.samplerate)

    @check_units(onset=second)
    def shift(self, onset):
        '''
        Returns the sound played at time onset.
        '''
        onset = int(onset*self.samplerate)
        y = vstack((zeros((onset, self.nchannels)), self))
        return Sound(y, rate=self.samplerate)

    def repeat(self, n):
        '''
        Repeats the sound n times
        '''
        x = vstack((self,)*n)
        return Sound(x, rate=self.samplerate)

    ### TODO: test this - I haven't installed scikits.samplerate on windows
    # it should work, according to the documentation 2D arrays are acceptable
    # in the format we use fof sounds here
    @check_units(rate=Hz)
    def resample(self, rate, resample_type='sinc_best'):
        '''
        Returns a resampled version of the sound.
        '''
        if not have_scikits_samplerate:
            raise ImportError('Need scikits.samplerate package for resampling')
        y = array(resample(self, float(rate / self.samplerate), resample_type),
                  dtype=float64)
        return Sound(y, rate=rate)

    def copy_from(self, other):
        '''
        Copies values from the given sound (resampled if necessary).
        '''
        if not isinstance(other, Sound):
            raise TypeError('Must copy from a Sound object.')
        if int(other.samplerate) != int(self.samplerate):
            other = other.resample(self.samplerate)
        self[:min(len(other), len(self)), :] = other[:min(len(other), len(self)), :]

    def play(self, normalise=False, sleep=False):
        '''
        Plays the sound (normalised to avoid clipping if required).
        '''
        if self.nchannels>2:
            raise ValueError("Can only play sounds with 1 or 2 channels.")
        pygame.mixer.quit()
        pygame.mixer.init(int(self.samplerate), -16, self.nchannels)
        if normalise:
            a = amax(abs(self))
        else:
            a = 1
        x = array((2 ** 15 - 1) * clip(self / a, -1, 1), dtype=int16)
        if self.nchannels==1:
            x.shape = x.size
        x = pygame.sndarray.make_sound(x)
        x.play()
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
        if self.nchannels>1:
            raise ValueError('Can only plot spectrograms for mono sounds.')
        x = self.flatten()
        pxx, freqs, bins, im = specgram(x, Fs=self.samplerate, **kwds)
        if frequency_range is not None:
            I = logical_and(frequency_range[0] <= freqs, freqs <= frequency_range[1])
            I2 = where(I)[0]
            I2 = [max(min(I2) - 1, 0), min(max(I2) + 1, len(freqs) - 1)]
            Z = pxx[I2[0]:I2[-1], :]
        else:
            Z = pxx
        if log_spectrogram:
            Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
            Z = 10 * log10(Z)
        Z = flipud(Z)
        if frequency_range is not None:
            imshow(Z, extent=(0, amax(bins), freqs[I2[0]], freqs[I2[-1]]), aspect='auto')
        else:
            imshow(Z, extent=(0, amax(bins), freqs[0], freqs[-1]), aspect='auto')
        xlabel('Time (s)')
        ylabel('Frequency (Hz)')
        return (pxx, freqs, bins, im)

    def spectrum(self, frequency_range=None, log_spectrum=True, display=False):
        '''
        Plots and returns the spectrum of the sound
        
        If frequency_range=None it shows the full spectrum, otherwise
        frequency_range=(minfreq, maxfreq).
        
        If log_spectrogram=True it shows log power, otherwise not.
        
        If display=True it plots the spectrum and phase, otherwise not.
        
        Returns the values Z, freqs, phase
        where Z is a 1D array of powers, freqs is the corresponding frequencies,
        phase if the unwrapped phase of spectrum.
        '''
        if self.nchannels>1:
            raise ValueError('Can only plot spectrum for mono sounds.')
        sp = numpy.fft.fft(array(self))
        freqs = array(range(len(sp)), dtype=float64) / len(sp) * float64(self.samplerate)
        pxx = abs(sp) ** 2
        phase = unwrap(mod(angle(sp), 2 * pi))
        if frequency_range is not None:
            I = logical_and(frequency_range[0] <= freqs, freqs <= frequency_range[1])
            I2 = where(I)[0]
            Z = pxx[I2]
            freqs = freqs[I2]
            phase = phase[I2]
        else:
            Z = pxx
        if log_spectrum:
            Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
            Z = 10 * log10(Z)
        if display:
            subplot(211)
            semilogx(freqs, Z)
            ticks_freqs = 32000 * 2 ** -array(range(18), dtype=float64)
            xticks(ticks_freqs, map(str, ticks_freqs))
            grid()
            xlim((freqs[0], freqs[-1]))
            xlabel('Frequency (Hz)')
            ylabel('Power (dB)') if log_spectrum else ylabel('Power')
            subplot(212)
            semilogx(freqs, phase)
            ticks_freqs = 32000 * 2 ** -array(range(18), dtype=float64)
            xticks(ticks_freqs, map(str, ticks_freqs))
            grid()
            xlim((freqs[0], freqs[-1]))
            xlabel('Frequency (Hz)')
            ylabel('Phase (rad)')
            show()
        return (Z, freqs, phase)

    def intensities(self, type='rms'):
        '''
        Returns intensity in dB SPL assuming array is in Pascals
        Returns an array of intensities for each channel, even if there is
        only one channel.
        '''
        if type=='rms':
            return 20.0*log10(sqrt(mean(asarray(self)**2, axis=0))/2e-5)
        elif type=='peak':
            # TODO: where does this number come from? Maybe we should make it
            # a named constant? Probably for 2e-5 above too (can't remember
            # where that comes from, but I'm sure it's very sensible).
            return 28e-6*10**(amax(asarray(self), axis=0)/20.)
        else:
            raise ValueError('Intensity type must be rms or peak')

    def intensity(self, type='rms'):
        '''
        Returns intensity in dB SPL assuming array is in Pascals
        In the case of multi-channel sounds, returns a tuple of intensities
        for each channel, otherwise returns a float.
        '''
        I = self.intensities(type=type)
        if self.nchannels==1:
            return I[0]
        else:
            return tuple(I) 
    
    ### TODO: update to use multiple channels
    def setintensity(self, dB, type='rms'): #replace atintensity
        '''
        Set the intensity (in dB) of a given signal
        the dB scale can be with respect to the rms value (default) or peak value
        by choosing type='rms' or type='peak'
        note: the signal should be long enough to measure the rms value
        '''
        # TODO: what does the comment "replace atintensity mean?" - does it mean
        # that there is a problem with the atintensity function? If so, let's
        # replace it with this one.
        # TODO: grammatical point: a method named setintensity should change the
        # values of this sound, not return a new one, which is why I called the
        # original one atintensity.
        if type=='rms':
            rms_value = sqrt(mean((asarray(self)-mean(asarray(self)))**2))
            rms_dB = 20.0*log10(rms_value/2e-5)
            gain = 10**((dB-rms_dB)/20.)
        elif type=='peak':
            # TODO: do you want to normalise here, or return a normalised
            # version of this? At the moment, this does nothing (see comments
            # for normalize function below).
            self.normalize()
            gain = 28e-6*10**(dB/20.)
        
        return self*gain
            
    def normalize(self):
        # TODO: what should this function do? As it is, it should be called
        # something like normalized because it doesn't change the sound itself,
        # it just returns a normalised version of it. So either the name
        # should change or the behaviour. We should probably have both. And
        # let's have both US and UK spellings too (normalise/normalize).
        factor = max(max(asarray(self)),abs(min(asarray(self))))
        return self/factor
        
    def atintensity(self, db):
        '''
        Returns sound in Pascals at various intensities (in RMS dB SPL)
        '''
        return self.amplified(db - self.intensity())

    # TODO: rename/remove?
    def amplified(self, db):
        '''
        Returns sound amplified by a given amount in dB pressure.
        '''
        return self * 10.0 ** (db / 20.0)

    def ramp(self, when='both', duration=10*ms, func=None, inplace=True):
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
        if func is None: func = lambda t:sin(pi * t / 2) ** 2
        sz = int(duration * self.samplerate)
        multiplier = func(reshape(linspace(0.0, 1.0, sz), (sz, 1)))
        if inplace:
            target = self
        else:
            target = Sound(copy(self), self.samplerate)
        if when == 'on' or when == 'both':
            target[:sz, :] *= multiplier
        if when == 'off' or when == 'both':
            target[-sz:, :] *= multiplier[::-1]
        return target
    
    def ramped(self, when='both', duration=10*ms, func=None, inplace=False):
        return self.ramp(when=when, duration=duration, func=func, inplace=inplace)

    @staticmethod
    def tone(freq, duration, rate=44.1*kHz, dB=None, dBtype='rms'):
        # TODO: do we want to include the dB and dBtype options here? I would
        # tend to say no because you can set the intensity yourself elsewhere,
        # and this duplicates the functionality?
        '''
        Returns a pure tone at the given frequency for the given duration
        if dB not given, pure tone is between -1 and 1
        '''
        x = sin(2.0*pi*freq*arange(0*ms, duration, 1/sound_rate))
        if dB is not None: 
            return Sound(x, rate).setintensity(dB, type=dBtype)
        else:
            return Sound(x, rate)

    @staticmethod
    def whitenoise(duration, rate=44.1*kHz, dB=None, dBtype='rms'):
        # TODO: same comment as for tone about dB/dBtype
        '''
        Returns a white noise for the given duration.
        if dB not given, white noise with a variance of one
        '''
        x = randn(int(rate*duration))
        
        if dB is not None: 
            return Sound(x, rate).setintensity(dB,type=dBtype)
        else:
            return Sound(x, rate)

    @staticmethod
    def click(duration, amplitude=1, rate=44.1*kHz, dB=None):
        # TODO: similar comment to tone/whitenoise
        '''
        Returns a click with given parameters
        if dB not given, click of amplitude given by the parameter amplitude
        note that the dB can only be peak dB SPL
        '''
        if dB is not None:
            amplitude = 28e-6*10**(dB/20.)
        
        x = amplitude*ones(int(duration*rate))
        return Sound(x, rate)

    @staticmethod
    def silent(duration, rate=44.1*kHz):
        '''
        Returns a silent, zero sound for the given duration.
        '''
        x = numpy.zeros(int(duration*rate))
        return Sound(x, rate)

    @staticmethod
    def sequence(sounds, rate=None):
        '''
        Returns the sequence of sounds in the list sounds joined together
        '''
        if rate is None:
            rate = max(s.samplerate for s in sounds)
            rates = unique([int(s.samplerate) for s in sounds])
            if len(rates)>1:
                sounds = tuple(s.resample(rate) for s in sounds)
        x = vstack(sounds)
        return Sound(x, rate)

    ### TODO: update to use multiple channels
    # TODO: remove the rate keyword from load? redundant.
    @staticmethod
    def load(filename, rate=rate):
        '''
        Currently, 8 and 16 bit mono sounds are handled correctly.
        Specify rate to resample to a given rate.
        '''
        forced_rate = rate
        f = wave.open(filename, 'r')
        sampwidth = f.getsampwidth()
        typecode = {2:'h', 1:'B'}[sampwidth]
        scale = {2:2 ** 14, 1:2 ** 7}[sampwidth]
        meanval = {2:0, 1:1}[sampwidth]
        rate = f.getframerate()
        x = f.readframes(f.getnframes())
        x = pyarray.array(typecode, x)
        x = array(x, dtype=float)
        x = x / scale
        x = x - meanval
        f.close()
        rate = rate * Hz
        if forced_rate is not None:
            rate, x = resample_sound(x, rate, forced_rate)
        return (rate, x)

    ### TODO: update to use multiple channels, without the stereo keyword
    def save(self, filename, normalise=False, sampwidth=2, stereo=None):
        '''
        Save the sound as a WAV file.
        
        If the normalise keyword is set to True, the amplitude of the sound will be
        normalised to 1. The sampwidth keyword can be 1 or 2 to save the data as
        8 or 16 bit samples. If the stereo keyword is set, it should be the
        right hand channel (and the original instance should be the left hand
        channel).
        '''
        if sampwidth != 1 and sampwidth != 2:
            raise ValueError('Sample width must be 1 or 2 bytes.')
        typecode = {2:'h', 1:'B'}[sampwidth]
        scale = {2:2 ** 14, 1:2 ** 7}[sampwidth]
        meanval = {2:0, 1:1}[sampwidth]
        dtype = {2:int16, 1:uint8}[sampwidth]
        w = wave.open(filename, 'wb')
        if stereo is None:
            w.setnchannels(1)
        else:
            w.setnchannels(2)
        w.setsampwidth(sampwidth)
        w.setframerate(int(self.samplerate))
        x = asarray(self)
        if normalise and stereo is None:
            x /= amax(x)
        if stereo is not None:
            y = asarray(stereo)
            if normalise:
                am = max(amax(x), amax(y))
                x /= am
                y /= am
            x = (x + meanval) * scale
            y = (y + meanval) * scale
            z = zeros(len(x) + len(y), dtype=x.dtype)
            z[0::2] = x
            z[1::2] = y
            data = array(z, dtype=dtype)
            data = pyarray.array(typecode, data)
            w.writeframes(data.tostring())
        else:
            x = (x + meanval) * scale
            data = array(x, dtype=dtype)
            data = pyarray.array(typecode, data)
            w.writeframes(data.tostring())
        w.close()

    def __reduce__(self):
        return (_load_Sound_from_pickle, (asarray(self), float(self.samplerate)))


def _load_Sound_from_pickle(arr, rate):
    return Sound(arr, rate=rate*Hz)


whitenoise = Sound.whitenoise
tone = Sound.tone
click = Sound.click
silent = Sound.silent
sequence = Sound.sequence
