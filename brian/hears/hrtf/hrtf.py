from brian import *
from ..sounds import Sound
from ..filtering import FIRFilterbank
from copy import copy

__all__ = ['HRTF', 'HRTFSet', 'HRTFDatabase',
           'make_coordinates']

class HRTF(object):
    '''
    Head related transfer function.
    
    **Attributes**

    ``impulse_response``
        The pair of impulse responses (as stereo :class:`Sound` objects)
    ``fir``
        The impulse responses in a format suitable for using with
        :class:`FIRFilterbank` (the transpose of ``impulse_response``).
    ``left``, ``right``
        The two HRTFs (mono :class:`Sound` objects)
    ``samplerate``
        The sample rate of the HRTFs.
        
    **Methods**
    
    .. automethod:: apply
    .. automethod:: filterbank
    
    You can get the number of samples in the impulse response with ``len(hrtf)``.        
    '''
    def __init__(self, hrir_l, hrir_r=None):
        if hrir_r is None:
            hrir = hrir_l
        else:
            hrir = Sound((hrir_l, hrir_r), samplerate=hrir_l.samplerate)
        self.samplerate = hrir.samplerate
        self.impulse_response = hrir
        self.left = hrir.left
        self.right = hrir.right

    def apply(self, sound):
        '''
        Returns a stereo :class:`Sound` object formed by applying the pair of
        HRTFs to the mono ``sound`` input. Equivalently, you can write
        ``hrtf(sound)`` for ``hrtf`` an :class:`HRTF` object.
        '''
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
        left_sound = ifft(left_sound_fft).real
        right_sound = ifft(right_sound_fft).real
        # finally, we take only the unpadded parts of these
        left_sound = left_sound[ir_nmax:ir_nmax+len(sound)]
        right_sound = right_sound[ir_nmax:ir_nmax+len(sound)]
        return Sound((left_sound, right_sound), samplerate=self.samplerate)        
    __call__ = apply

    def get_fir(self):
        return array(self.impulse_response.T, copy=True)
    fir = property(fget=get_fir)

    def filterbank(self, source, **kwds):
        '''
        Returns an :class:`FIRFilterbank` object that can be used to apply
        the HRTF as part of a chain of filterbanks.
        '''
        return FIRFilterbank(source, self.fir, **kwds)
    
    def __len__(self):
        return self.impulse_response.shape[0]

def make_coordinates(**kwds):
    '''
    Creates a numpy record array from the keywords passed to the function.
    Each keyword/value pair should be the name of the coordinate the array of
    values of that coordinate for each location.
    Returns a numpy record array. For example::
    
        coords = make_coordinates(azimuth=[0, 30, 60, 0, 30, 60],
                                  elevation=[0, 0, 0, 30, 30, 30])
        print coords['azimuth']
    '''
    dtype = [(name, float) for name in kwds.keys()]
    n = len(kwds.values()[0])
    x = zeros(n, dtype=dtype)
    for name, values in kwds.items():
        x[name] = values
    return x

class HRTFSet(object):
    '''
    A collection of HRTFs, typically for a single individual.
    
    Normally this object is created automatically by an :class:`HRTFDatabase`.
        
    **Attributes**
    
    ``hrtf``
        A list of ``HRTF`` objects for each index.
    ``num_indices``
        The number of HRTF locations. You can also use ``len(hrtfset)``.
    ``num_samples``
        The sample length of each HRTF.
    ``fir_serial``, ``fir_interleaved``
        The impulse responses in a format suitable for using with
        :class:`FIRFilterbank`, in serial (LLLLL...RRRRR....) or interleaved
        (LRLRLR...).
    
    **Methods**
    
    .. automethod:: subset
    .. automethod:: filterbank
    
    You can access an HRTF by index via ``hrtfset[index]``, or
    by its coordinates via ``hrtfset(coord1=val1, coord2=val2)``.
    
    **Initialisation**
    
    ``data``
        An array of shape (2, num_indices, num_samples) where data[0,:,:] is
        the left ear and data[1,:,:] is the right ear, num_indices is the number
        of HRTFs for each ear, and num_samples is the length of the HRTF.
    ``samplerate``
        The sample rate for the HRTFs (should have units of Hz).
    ``coordinates``
        A record array of length ``num_indices`` giving the coordinates of each
        HRTF. You can use :func:`make_coordinates` to help with this.
    '''
    def __init__(self, data, samplerate, coordinates):
        self.data = data
        self.samplerate = samplerate
        self.coordinates = coordinates
        self.hrtf = []
        for i in xrange(self.num_indices):
            l = Sound(self.data[0, i, :], samplerate=self.samplerate)
            r = Sound(self.data[1, i, :], samplerate=self.samplerate)
            self.hrtf.append(HRTF(l, r))
            
    def __getitem__(self, key):
        return self.hrtf[key]
    
    def __call__(self, **kwds):
        I = ones(self.num_indices, dtype=bool)
        for key, value in kwds.items():
            I = logical_and(I, abs(self.coordinates[key]-value)<1e-10)
        indices = I.nonzero()[0]
        if len(indices)==0:
            raise IndexError('No HRTF exists with those coordinates')
        if len(indices)>1:
            raise IndexError('More than one HRTF exists with those coordinates')
        return self.hrtf[indices[0]]

    def subset(self, condition):
        '''
        Generates the subset of the set of HRTFs whose coordinates satisfy
        the ``condition``. This should be one of: a boolean array of
        length the number of HRTFs in the set, with values
        of True/False to indicate if the corresponding HRTF should be included
        or not; an integer array with the indices of the HRTFs to keep; or a
        function whose argument names are
        names of the parameters of the coordinate system, e.g.
        ``condition=lambda azim:azim<pi/2``.
        '''
        if callable(condition):
            ns = dict((name, self.coordinates[name]) for name in condition.func_code.co_varnames)
            try:
                I = condition(**ns)
                I = I.nonzero()[0]
            except:
                I = False
            if isinstance(I, bool): # vector-based calculation doesn't work
                n = len(ns[condition.func_code.co_varnames[0]])
                I = array([condition(**dict((name, ns[name][j]) for name in condition.func_code.co_varnames)) for j in range(n)])
                I = I.nonzero()[0]
        else:
            if condition.dtype==bool:
                I = condition.nonzero()[0]
            else:
                I = condition
        hrtf = [self.hrtf[i] for i in I]
        coords = self.coordinates[I]
        data = self.data[:, I, :]
        obj = copy(self)
        obj.hrtf = hrtf
        obj.coordinates = coords
        obj.data = data
        return obj
    
    def __len__(self):
        return self.num_indices
    
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
        '''
        Returns an :class:`FIRFilterbank` object which applies all of the HRTFs
        in the set. If ``interleaved=False`` then
        the channels are arranged in the order LLLL...RRRR..., otherwise they
        are arranged in the order LRLRLR....
        '''
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
    '''
    def __init__(self, samplerate=None):
        raise NotImplementedError

    def load_subject(self, subject):
        raise NotImplementedError
