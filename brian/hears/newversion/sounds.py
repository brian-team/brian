from brian import *
from numpy import *
import numpy
import array as pyarray
import time
import struct
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
from prefs import get_samplerate
from db import dB, dB_type, dB_error, gain

__all__ = ['BaseSound', 'Sound',
           'pinknoise','brownnoise','powerlawnoise',
           'whitenoise', 'tone', 'click', 'clicks', 'silence', 'sequence',
           'loadsound', 'savesound', 'playsound',
           ]

class BaseSound(Bufferable):
    '''
    Base class for Sound and OnlineSound
    '''
    pass

class Sound(BaseSound, numpy.ndarray):
    '''
    TODO: documentation for Sound

    Slicing:
       One can slice sound objects in various ways:
    ``s[start:stop:step,channel]``
       Returns another Sound object that corresponds to the slice imposed. 
       ``start`` and ``stop`` values can be specified in samples or durations (e.g. 100*ms). 
       ``step`` is only implemented for integer values (no resampling). 
       Still one can use step to reverse the signal (e.g. using s[100*ms:50*ms:-1]).
       ``channel`` can be a slice for multi channel sounds, that is s[10*ms:50*ms,:] is equivalent to s[10*ms:50*ms].
       One can also specify start and stop values outside the Sound range (start<0 or stop> duration), in this case the Sound is zero-padded to fit the desired duration.
       
    '''
    duration = property(fget=lambda self:len(self) / self.samplerate)
    times = property(fget=lambda self:arange(len(self), dtype=float) / self.samplerate)
    nchannels = property(fget=lambda self:self.shape[1])
    left = property(fget=lambda self:self.channel(0))
    right = property(fget=lambda self:self.channel(1))

    @check_units(samplerate=Hz, duration=second)
    def __new__(cls, data, samplerate=None, duration=None):
        if isinstance(data, numpy.ndarray):
            samplerate = get_samplerate(samplerate)
#            if samplerate is None:
#                raise ValueError('Must specify samplerate to initialise Sound with array.')
            if duration is not None:
                raise ValueError('Cannot specify duration when initialising Sound with array.')
            x = array(data, dtype=float)
        elif isinstance(data, str):
            if duration is not None:
                raise ValueError('Cannot specify duration when initialising Sound from file.')
            x = Sound.load(data, samplerate=samplerate)
            samplerate = x.samplerate
        elif callable(data):
            samplerate = get_samplerate(samplerate)
#            if samplerate is None:
#                raise ValueError('Must specify samplerate to initialise Sound with function.')
            if duration is None:
                raise ValueError('Must specify duration to initialise Sound with function.')
            L = int(duration * samplerate)
            t = arange(L, dtype=float) / samplerate
            x = data(t)
        elif isinstance(data, (list, tuple)):
            kwds = {}
            if samplerate is not None:
                kwds['samplerate'] = samplerate
            if duration is not None:
                kwds['duration'] = duration
            channels = tuple(Sound(c, **kwds) for c in data)
            x = hstack(channels)
            samplerate = channels[0].samplerate
        else:
            raise TypeError('Cannot initialise Sound with data of class ' + str(data.__class__))
        if len(x.shape)==1:
            x.shape = (len(x), 1)
        x = x.view(cls)
        x.samplerate = samplerate
        x.buffer_init()
        return x

    def __array_wrap__(self, obj, context=None):
        handled = False
        x = numpy.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'samplerate') and hasattr(self, 'samplerate'):
            x.samplerate = self.samplerate
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x
    
    def __array_finalize__(self,obj):
        if obj is None: return
        self.samplerate = getattr(obj, 'samplerate', None)
        
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
                other = other.resized(len(self))
            elif len(self) < len(other):
                self = self.resized(len(other))

            return Sound(numpy.ndarray.__add__(self, other), samplerate=self.samplerate)
        else:
            x = numpy.ndarray.__add__(self, other)
            return Sound(x, self.samplerate)
    __radd__ = __add__


    def __rshift__(self,other):
        return Sound(vstack((self,other)))
    
    def __lshift__(self,other):
        return Sound(vstack((self,other)))

    def __getitem__(self,key):
        channel=slice(None)
        if isinstance(key,tuple):
            channel=key[1]
            key=key[0]

        if isinstance(key,int):
            return np.ndarray.__getitem__(self,key)
        if isinstance(key,Quantity):
            return np.ndarray.__getitem__(self,round(key*self.samplerate))

        sliceattr = [v for v in [key.start, key.stop] if v is not None]
        slicedims = array([units.have_same_dimensions(flag,second) for flag in sliceattr])

        if not slicedims.any():
            start = key.start or 0
            stop = key.stop or self.shape[0]
            step = key.step or 1
            if start >= 0 and stop <= self.shape[0]:
                return Sound(np.ndarray.__getitem__(self,(key,channel)),self.samplerate)
            else:
                bpad = Sound(zeros((-start,self.shape[1])))
                apad = Sound(zeros((stop-self.shape[0],self.shape[1])))
                if step==-1:
                    return Sound(vstack((apad,flipud(asarray(self)),bpad)),self.samplerate)
                return Sound(vstack((bpad,asarray(self),apad)),self.samplerate)
        if not slicedims.all():
            raise DimensionMismatchError('Slicing',*[units.get_unit(d) for d in sliceattr])
        
        start = key.start or 0*msecond
        stop = key.stop or self.duration
        step = key.step or 1
        if int(step)!=step:
            #resampling
            raise NotImplementedError
        start = int(start*self.samplerate)
        stop = int(stop*self.samplerate)
        return self.__getitem__((slice(start,stop,step),channel))
    
    def __setitem__(self,key,value):
        channel=slice(None)
        if isinstance(key,tuple):
            channel=key[1]
            key=key[0]
        
        if isinstance(key,int) or isinstance(key,float):
            return np.ndarray.__setitem__(self,(key,channel),value)
        if isinstance(key,Quantity):
            return np.ndarray.__setitem__(self,(round(key*self.samplerate),channel),value)

        sliceattr = [v for v in [key.start, key.step, key.stop] if v is not None]
        slicedims=array([units.have_same_dimensions(flag,second) for flag in sliceattr])
        if not slicedims.any():
            # If value is a mono sound its shape will be (N, 1) but the numpy
            # setitem will have shape (N,) so in this case it's a shape mismatch
            # so we squeeze the arrya to make sure this doesn't happen.
            if isinstance(value,Sound) and channel!=slice(None):
                value=value.squeeze()
            return asarray(self).__setitem__((key,channel),value)
#            print value.shape
#            print self.shape
#            print np.ndarray.__getitem__(self, (key, channel)).shape
#            print key, channel
#            return np.ndarray.__setitem__(self, (key, channel), value)
        if not slicedims.all():
            raise DimensionMismatchError('Slicing',*[units.get_unit(d) for d in sliceattr])

        if key.__getattribute__('step') is not None:
            # resampling?
            raise NotImplementedError
        
        start = key.start
        stop = key.stop or self.duration
        if (start is not None and start<0*ms) or stop > self.duration:
            raise IndexError('Slice bigger than Sound object')
        if start is not None: start = int(round(start*self.samplerate))
        stop = int(stop*self.samplerate)
        return self.__setitem__((slice(start,stop),channel),value)

    @check_units(duration=second)
    def extended(self, duration):
        '''
        Returns the Sound with length extended by the given duration.
        '''
        if not isinstance(duration, int):
            duration = int(duration * self.samplerate)
        return self[:self.duration+duration]

    def resized(self, L):
        '''
        Returns the Sound with length extended (or contracted) to have L samples.
        '''
        if L == len(self):
            return self
        elif L < len(self):
            return Sound(self[:L, :], samplerate=self.samplerate)
        else:
            padding = zeros((L - len(self), self.nchannels))
            return Sound(concatenate((self, padding)), samplerate=self.samplerate)

    @check_units(duration=second)
    def shift(self, duration):
        '''
        Returns the sound delayed by duration.
        '''
        if not isinstance(duration, int):
            duration = int(duration*self.samplerate)
        y = vstack((zeros((duration, self.nchannels)), self))
        return Sound(y, samplerate=self.samplerate)

    def repeat(self, n):
        '''
        Repeats the sound n times
        '''
        x = vstack((self,)*n)
        return Sound(x, samplerate=self.samplerate)

    ### TODO: test this - I haven't installed scikits.samplerate on windows
    # it should work, according to the documentation 2D arrays are acceptable
    # in the format we use fof sounds here
    @check_units(samplerate=Hz)
    def resample(self, samplerate, resample_type='sinc_best'):
        '''
        Returns a resampled version of the sound.
        '''
        if not have_scikits_samplerate:
            raise ImportError('Need scikits.samplerate package for resampling')
        y = array(resample(self, float(samplerate / self.samplerate), resample_type),
                  dtype=float64)
        return Sound(y, samplerate=samplerate)

    def play(self, normalise=False, sleep=False):
        '''
        Plays the sound (normalised to avoid clipping if required). If
        sleep=True then the function will wait until the sound has finished
        playing before returning.
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

    def get_level(self):
        '''
        Returns level in dB SPL (RMS) assuming array is in Pascals.
        In the case of multi-channel sounds, returns an array of levels
        for each channel, otherwise returns a float.
        '''
        if self.nchannels==1:
            rms_value = sqrt(mean((asarray(self)-mean(asarray(self)))**2))
            rms_dB = 20.0*log10(rms_value/2e-5)
            return rms_dB*dB
        else:
            return array(tuple(self.channel(i).get_level() for i in xrange(self.nchannels)))

    def set_level(self, level):
        '''
        Sets level in dB SPL (RMS) assuming array is in Pascals. ``level``
        should be a value in dB, or a tuple of levels, one for each channel.
        '''
        rms_dB = self.get_level()
        if self.nchannels>1:
            level = array(level)
            if level.size==1:
                level = level.repeat(self.nchannels)
            level = reshape(level, (1, self.nchannels))
            rms_dB = reshape(rms_dB, (1, self.nchannels))
        else:
            if not isinstance(level, dB_type):
                raise dB_error('Must specify level in dB')
            rms_dB = float(rms_dB)
            level = float(level)
        gain = 10**((level-rms_dB)/20.)
        self *= gain

    level = property(fget=get_level, fset=set_level)
    
    def atlevel(self, level):
        '''
        Returns the sound at the given level in dB SPL (RMS) assuming array is
        in Pascals. ``level`` should be a value in dB, or a tuple of levels,
        one for each channel.
        '''
        newsound = self.copy()
        newsound.level = level
        return newsound
            
    def ramp(self, when='onset', duration=10*ms, envelope=None, inplace=True):
        '''
        Adds a ramp on/off to the sound
        
        ``when='onset'``
            Can take values 'onset', 'offset' or 'both'
        ``duration=10*ms``
            The time over which the ramping happens
        ``envelope``
            A ramping function, if not specified uses ``sin(pi*t/2)**2``. The
            function should be a function of one variable ``t`` ranging from
            0 to 1, and should increase from ``f(0)=0`` to ``f(0)=1``. The
            reverse is applied for the offset ramp.
        ``inplace``
            Whether to apply ramping to current sound or return a new array.
        '''
        when = when.lower().strip()
        if envelope is None: envelope = lambda t:sin(pi * t / 2) ** 2
        if not isinstance(duration, int):
            sz = int(duration * self.samplerate)
        else:
            sz = duration
        multiplier = envelope(reshape(linspace(0.0, 1.0, sz), (sz, 1)))
        if inplace:
            target = self
        else:
            target = Sound(copy(self), self.samplerate)
        if when == 'onset' or when == 'both':
            target[:sz, :] *= multiplier
        if when == 'offset' or when == 'both':
            target[-sz:, :] *= multiplier[::-1]
        return target
    
    def ramped(self, when='onset', duration=10*ms, envelope=None):
        '''
        Returns a ramped version of the sound (see :meth:`Sound.ramp`).
        '''
        return self.ramp(when=when, duration=duration, envelope=envelope, inplace=False)

    @staticmethod
    def tone(frequency, duration, samplerate=None):
        '''
        Returns a pure tone at frequency for duration, using the default
        samplerate or the given one.
        '''
        samplerate = get_samplerate(samplerate)
        x = sin(2.0*pi*freq*arange(0*ms, duration, 1/samplerate))
        return Sound(x, samplerate)

    @staticmethod
    def whitenoise(duration, samplerate=None):
        '''
        Returns a white noise. If the samplerate is not specified, the global
        default value will be used.
        '''
        samplerate = get_samplerate(samplerate)
        x = randn(int(samplerate*duration))
        return Sound(x, samplerate)

    @staticmethod
    def powerlawnoise(duration, alpha, samplerate=None):
        '''
        Returns a power-law noise for the given duration. Spectral density per unit of bandwidth scales as 1/(f**alpha).
        
        Sample usage::
        
            noise = powerlawnoise(200*ms, 1, samplerate=44100*Hz)
        
        Arguments:
        
        ``duration`` 
            Duration of the desired output.
        ``alpha``
            Power law exponent.
        ``samplerate``
            Desired output samplerate
        '''
        samplerate = get_samplerate(samplerate)
        # Adapted from http:/+/www.eng.ox.ac.uk/samp/software/powernoise/powernoise.m
        # Little MA et al. (2007), "Exploiting nonlinear recurrence and fractal
        # scaling properties for voice disorder detection", Biomed Eng Online, 6:23
        n=duration*samplerate
        n2=floor(n/2)
        f=fftfreq(int(n),d=1*second/samplerate)
    
        a2=1/(f[1:n2]**(alpha/2))
        p2=(rand(n2-1)-0.5)*2*pi
        d2=a2*exp(1j*p2)
        
        d=hstack((1,d2,1/f[-1]**alpha,flipud(conj(d2))))
        
        x=real(ifft(d))
        x.shape=(n,1)
        x = ((x - min(x))/(max(x) - min(x)) - 0.5) * 2;
        return Sound(x,samplerate)
    
    @staticmethod
    def pinknoise(duration, samplerate=None):
        '''
        Returns pink noise, i.e :func:`powerlawnoise` with alpha=1
        '''
        return Sound.powerlawnoise(duration,1,samplerate=samplerate)
    
    @staticmethod
    def brownnoise(duration, samplerate=None):
        '''
        Returns brown noise, i.e :func:`powerlawnoise` with alpha=2
        '''
        return Sound.powerlawnoise(duration,2,samplerate=samplerate)

    @staticmethod
    def click(duration, peak=None, samplerate=None):
        '''
        Returns a click of the given duration.
        
        If ``peak`` is not specified, the amplitude will be 1, otherwise
        ``peak`` refers to the peak dB SPL of the click, according to the
        formula ``28e-6*10**(peak/20.)``.
        '''
        samplerate = get_samplerate(samplerate)
        if peak is not None:
            if not isinstance(peak, dB_type):
                raise dB_error('Peak must be given in dB')
            amplitude = 28e-6*10**(peak/20.)
        x = amplitude*ones(int(duration*samplerate))
        return Sound(x, samplerate)
    
    @staticmethod
    def clicks(duration, n, interval, peak=None, samplerate=None):
        '''
        Returns a series of n clicks (see :func:`click`) separated by interval.
        '''
        oneclick = Sound.click(duration, peak=peak, samplerate=samplerate)
        return oneclick[:interval].repeat(n)

    @staticmethod
    def silence(duration, samplerate=None):
        '''
        Returns a silent, zero sound for the given duration.
        '''
        samplerate = get_samplerate(samplerate)
        x = numpy.zeros(int(duration*samplerate))
        return Sound(x, samplerate)

    @staticmethod
    def sequence(sounds, samplerate=None):
        '''
        Returns the sequence of sounds in the list sounds joined together
        '''
        if samplerate is None:
            samplerate = max(s.samplerate for s in sounds)
            rates = unique([int(s.samplerate) for s in sounds])
            if len(rates)>1:
                sounds = tuple(s.resample(samplerate) for s in sounds)
        x = vstack(sounds)
        return Sound(x, samplerate)

    def save(self, filename, normalise=False, samplewidth=2):
        '''
        Save the sound as a WAV.
        
        If the normalise keyword is set to True, the amplitude of the sound will be
        normalised to 1. The samplewidth keyword can be 1 or 2 to save the data as
        8 or 16 bit samples.
        '''
        ext=filename.split('.')[-1]
        if ext=='wav':
            import wave as sndmodule
        elif ext=='aiff' or ext=='aifc':
            import aifc as sndmodule
            raise NotImplementedError('Can only save as wav soundfiles')
        else:
            raise NotImplementedError('Can only save as wav soundfiles')
        
        if samplewidth != 1 and samplewidth != 2:
            raise ValueError('Sample width must be 1 or 2 bytes.')
        
        scale = {2:2 ** 15, 1:2 ** 7-1}[samplewidth]
        if ext=='wav':
            meanval = {2:0, 1:2**7}[samplewidth]
            dtype = {2:int16, 1:uint8}[samplewidth]
            typecode = {2:'h', 1:'B'}[samplewidth]
        else:
            meanval = {2:0, 1:2**7}[samplewidth]
            dtype = {2:int16, 1:uint8}[samplewidth]
            typecode = {2:'h', 1:'B'}[samplewidth]
        w = sndmodule.open(filename, 'wb')
        w.setnchannels(self.nchannels)
        w.setsampwidth(samplewidth)
        w.setframerate(int(self.samplerate))
        x = array(self,copy=True)
        am=amax(x)
        z = zeros(x.shape[0]*self.nchannels, dtype=x.dtype)
        x.shape=(x.shape[0],self.nchannels)
        for i in range(self.nchannels):
            if normalise:
                x[:,i] /= am
            x[:,i] = (x[:,i]) * scale + meanval
            z[i::self.nchannels] = x[::1,i]
        data = array(z, dtype=dtype)
        data = pyarray.array(typecode, data)
        w.writeframes(data.tostring())
        w.close()
    
    @staticmethod
    def load(filename):
        '''
        Load the file given by filename and returns a Sound object. 
        Sound file can be either a .wav or a .aif file.
        '''
        ext=filename.split('.')[-1]
        if ext=='wav':
            import wave as sndmodule
        elif ext=='aif' or ext=='aiff':
            import aifc as sndmodule
        else:
            raise NotImplementedError('Can only load aif or wav soundfiles')
        def everyOther (v, offset=0, channels=2):
            return [v[i] for i in range(offset, len(v), channels)]
        wav = sndmodule.open (filename, "r")
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
        frames = wav.readframes (nframes * nchannels)
        typecode = {2:'h', 1:'B'}[sampwidth]
        out = struct.unpack_from ("%d%s" % (nframes * nchannels,typecode), frames)
        scale = {2:2 ** 15, 1:2 ** 7-1}[sampwidth]
        meanval = {2:0, 1:2**7}[sampwidth]
        
        data=zeros((nframes,nchannels))
        for i in range(nchannels):
            data[:,i]=array(everyOther(out,offset=i,channels=nchannels))
            data[:,i]/=scale
            data[:,i]-=meanval
        return Sound(data,samplerate=framerate*Hz)

    def __reduce__(self):
        return (_load_Sound_from_pickle, (asarray(self), float(self.samplerate)))


def _load_Sound_from_pickle(arr, samplerate):
    return Sound(arr, samplerate=samplerate*Hz)

def playsound(sound, normalise=False, sleep=False):
    sound.play(normalise=normalise, sleep=sleep)
playsound.__doc__ = Sound.play.__doc__

def savesound(sound, filename, normalise=False, samplewidth=2):
    sound.save(filename, normalise=normalise, samplewidth=samplewidth)
savesound.__doc__ = Sound.save.__doc__

whitenoise = Sound.whitenoise
powerlawnoise = Sound.powerlawnoise
pinknoise = Sound.pinknoise
brownnoise = Sound.brownnoise
tone = Sound.tone
click = Sound.click
clicks = Sound.clicks
silence = Sound.silence
sequence = Sound.sequence
loadsound = Sound.load
