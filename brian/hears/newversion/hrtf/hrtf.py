from brian import *
from ..sounds import *
from ..filtering import FIRFilterbank
from copy import copy

__all__ = ['HRTF', 'HRTFSet', 'HRTFDatabase']

class HRTF(object):
    '''
    HRTF class
    
    Has attributes:
    
    ``impulseresponse``
        The pair of impulse responses (as stereo :class:`Sound` objects)
    ``fir``
        The impulse responses in a format suitable for using with
        :class:`FIRFilterbank` (the transpose of ``impulseresponse``).
    ``left``, ``right``
        The two HRTFs (mono :class:`Sound` objects)
    ``samplerate``
        The sample rate of the HRTFs.
        
    Has methods:
    
    ``apply(sound)``
        Returns a stereo :class:`Sound` object formed by applying the pair of
        HRTFs to the mono ``sound`` input.
        
    ``filterbank(source, **kwds)``
        Returns an :class:`FIRFilterbank` object.
    '''
    def __init__(self, hrir_l, hrir_r=None):
        if hrir_r is None:
            hrir = hrir_l
        else:
            hrir = Sound((hrir_l, hrir_r), samplerate=hrir_l.samplerate)
        self.samplerate = hrir.samplerate
        self.impulseresponse = hrir
        self.left = hrir.left
        self.right = hrir.right

    def apply(self, sound):
        # Note we use an FFT based method for applying HRTFs that is
        # mathematically equivalent to using convolution (accurate to 1e-15
        # in practice) and around 100x faster.
        if not sound.nchannels==1:
            raise ValueError('HRTF can only be applied to mono sounds')
        if len(unique(array([self.samplerate, sound.samplerate], dtype=int)))>1:
            raise ValueError('HRTF and sound samplerates do not match.')
        sound = asarray(sound).flatten()
        # Pad left/right/sound with zeros of length max(impulse response length)
        # at the beginning, and at the end so that they are all the same length
        # which should be a power of 2 for efficiency. The reason to pad at
        # the beginning is that the first output samples are not guaranteed to
        # be equal because of the delays in the impulse response, but they
        # exactly equalise after the length of the impulse response, so we just
        # zero pad. The reason for padding at the end is so that for the FFT we
        # can just multiply the arrays, which should have the same shape.
        left = asarray(self.left).flatten()
        right = asarray(self.right).flatten()
        ir_nmax = max(len(left), len(right))
        nmax = max(ir_nmax, len(sound))+ir_nmax
        nmax = 2**int(ceil(log2(nmax)))
        leftpad = hstack((left, zeros(nmax-len(left))))
        rightpad = hstack((right, zeros(nmax-len(right))))
        soundpad = hstack((zeros(ir_nmax), sound, zeros(nmax-ir_nmax-len(sound))))
        # Compute FFTs, multiply and compute IFFT
        left_fft = fft(leftpad, n=nmax)
        right_fft = fft(rightpad, n=nmax)
        sound_fft = fft(soundpad, n=nmax)
        left_sound_fft = left_fft*sound_fft
        right_sound_fft = right_fft*sound_fft
        left_sound = ifft(left_sound_fft)
        right_sound = ifft(right_sound_fft)
        # finally, we take only the unpadded parts of these
        left_sound = left_sound[ir_nmax:ir_nmax+len(sound)]
        right_sound = right_sound[ir_nmax:ir_nmax+len(sound)]
        return Sound((left_sound, right_sound), samplerate=self.samplerate)        

    def get_fir(self):
        return array(self.impulseresponse.T, copy=True)
    fir = property(fget=get_fir)

    def filterbank(self, source, **kwds):
        return FIRFilterbank(source, self.fir, **kwds)

class HRTFSet(object):
    '''
    Base class for a collection of HRTFs for one individual
    
    Should have attributes:
    
    ``name``
        A unique string identifying this individual.
    ``data``
        An array of shape (2, num_indices, num_samples) where data[0,:,:] is
        the left ear and data[1,:,:] is the right ear, num_indices is the number
        of HRTFs for each ear, and num_samples is the length of the HRTF.
    ``samplerate``
        The sample rate for the HRTFs (should have units of Hz).
    ``coordinates``
        The record array of length num_indices of coordinates.
    
    Derived classes should override the ``load(...)`` method which should create
    the attributes above. The ``load`` method should have the following optional
    keywords:
    
    ``samplerate``
        The intended samplerate (resampling will be used if it is wrong). If
        left unset, the natural samplerate of the data set will be used.        
    ``coordsys``
        The intended coordinate system (conversion will be performed if it is
        different).
    
    Automatically generates the attributes:
    
    ``hrtf``
        A list of ``HRTF`` objects for each index.
    ``num_indices``
        The number of HRTF locations.
    ``num_samples``
        The sample length of each HRTF.
    ``fir_serial``, ``fir_interleaved``
        The impulse responses in a format suitable for using with
        :class:`FIRFilterbank`, in serial (LLLLL...RRRRR....) or interleaved
        (LRLRLR...).
    
    Has methods:
    
    ``subset(cond)``
        Generates the subset of the set of HRTFs whose coordinates satisfy
        the condition cond. cond should be a function whose argument names are
        names of the parameters of the coordinate system, e.g. for AzimElev you
        might do cond=lambda azim:azim<pi/2.
        
    ``filterbank(source, interleaved=False, **kwds)``
        Returns an :class:`FIRFilterbank` object. If ``interleaved=False`` then
        the channels are arranged in the order LLLL...RRRR..., otherwise they
        are arranged in the order LRLRLR....
    '''
    def __init__(self, *args, **kwds):
        self.load(*args, **kwds)
        self.prepare()

    def load(self, *args, **kwds):
        raise NotImplementedError

    def prepare(self):
        L, R = self.data
        self.hrtf = []
        for i in xrange(self.num_indices):
            l = Sound(self.data[0, i, :], samplerate=self.samplerate)
            r = Sound(self.data[1, i, :], samplerate=self.samplerate)
            self.hrtf.append(HRTF(l, r))

    def subset(self, cond):
        ns = dict((name, self.coordinates[name]) for name in cond.func_code.co_varnames)
        try:
            I = cond(**ns)
            I = I.nonzero()[0]
        except:
            I = False
        if type(I) == type(True): # vector-based calculation doesn't work
            n = len(ns[cond.func_code.co_varnames[0]])
            I = array([cond(**dict((name, ns[name][j]) for name in cond.func_code.co_varnames)) for j in range(n)])
            I = I.nonzero()[0]
        hrtf = [self.hrtf[i] for i in I]
        coords = self.coordinates[I]
        data = self.data[:, I, :]
        obj = copy(self)
        obj.hrtf = hrtf
        obj.coordinates = coords
        obj.data = data
        return obj
    
    @property
    def num_indices(self):
        return self.data.shape[1]
    
    @property
    def num_samples(self):
        return self.data.shape[2]
    
    @property
    def fir_serial(self):
        return reshape(self.data, (self.num_indices*2, self.num_samples))
    
    @property
    def fir_interleaved(self):
        fir = empty((self.num_indices*2, self.num_samples))
        fir[::2, :] = self.data[0, :, :]
        fir[1::2, :] = self.data[1, :, :]
        return fir
    
    def filterbank(self, source, interleaved=False, **kwds):
        if interleaved:
            fir = self.fir_interleaved
        else:
            fir = self.fir_serial
        return FIRFilterbank(source, fir, **kwds)


class HRTFDatabase(object):
    '''
    Base class for databases of HRTFs
    
    Should have an attribute 'subjects' giving a list of available subjects,
    and a method ``load_subject(subject)`` which returns an ``HRTFSet`` for that
    subject.
    
    The initialiser should take (optional) keywords:
    
    ``samplerate``
        The intended samplerate (resampling will be used if it is wrong). If
        left unset, the natural samplerate of the data set will be used.
    ``coordsys``
        The intended coordinate system (conversion will be performed if it is
        different).
    
    Should have a method:
    
    ``subject_name(subject)``
        Which returns a unique string id for the database and subject within
        the database.
    '''
    def __init__(self, samplerate=None, coordsys=None):
        raise NotImplementedError

    def load_subject(self, subject):
        raise NotImplementedError
