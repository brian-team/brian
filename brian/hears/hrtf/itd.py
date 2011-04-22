from brian import *
from hrtf import *
from ..prefs import get_samplerate
from ..filtering.fractionaldelay import FractionalDelay
from ..sounds import silence

__all__ = ['HeadlessDatabase']

speed_of_sound_in_air = 343.2*metre/second

class HeadlessDatabase(HRTFDatabase):
    '''
    Database for creating HRTFSet with artificial interaural time-differences
    
    Initialisation keywords:
    
    ``n``, ``azim_max``, ``diameter``
        Specify the ITDs for two ears separated by distance ``diameter`` with
        no head. ITDs corresponding to ``n`` angles equally spaced between
        ``-azim_max`` and ``azim_max`` are used. The default diameter is that
        which gives the maximum ITD as 650 microseconds. The ITDs are computed
        with the formula ``diameter*sin(azim)/speed_of_sound_in_air``. In this
        case, the generated :class:`HRTFSet` will have coordinates of ``azim``
        and ``itd``.
    ``itd``
        Instead of specifying the keywords above, just give the ITDs directly.
        In this case, the generated :class:`HRTFSet` will have coordinates of
        ``itd`` only.
    ``fractional_itds=False``
        Set this to ``True`` to allow ITDs with a fractional multiple of the
        timestep ``1/samplerate``. Note that the filters used to do this are
        not perfect and so this will introduce a small amount of numerical
        error, and so shouldn't be used unless this level of timing precision
        is required. See :class:`FractionalDelay` for more details.
        
    To get the HRTFSet, the simplest thing to do is just::
    
        hrtfset = HeadlessDatabase(13).load_subject()
        
    The generated ITDs can be returned using the ``itd`` attribute of the
    :class:`HeadlessDatabase` object.
    
    If ``fractional_itds=False`` then    
    Note that the delays induced in the left and right channels are not
    symmetric as making them so wastes half the samplerate (if the delay to
    the left channel is itd/2 and the delay to the right channel is -itd/2).
    Instead, for each channel either the left channel delay is 0 and the right
    channel delay is -itd (if itd<0) or the left channel delay is itd and the
    right channel delay is 0 (if itd>0).
    
    If ``fractional_itds=True`` then delays in the left and right channels will
    be symmetric around a global offset of ``delay_offset``.
    '''
    def __init__(self, n=None, azim_max=pi/2,
                 diameter=speed_of_sound_in_air*650*usecond,
                 itd=None, samplerate=None, fractional_itds=False):
        if itd is None:
            azim = linspace(-azim_max, azim_max, n)
            itd = diameter*sin(azim)/speed_of_sound_in_air
            coords = make_coordinates(azim=azim, itd=itd)
        else:
            coords = make_coordinates(itd=itd)
        self.itd = itd
        samplerate = self.samplerate = get_samplerate(samplerate)
        if not fractional_itds:
            dl = itd.copy()
            dr = -itd
            dl[dl<0] = 0
            dr[dr<0] = 0
            dl = array(rint(dl*samplerate), dtype=int)
            dr = array(rint(dr*samplerate), dtype=int)
            idxmax = max(amax(dl), amax(dr))
            data = zeros((2, len(itd), idxmax+1))
            data[0, arange(len(itd)), dl] = 1
            data[1, arange(len(itd)), dr] = 1
        else:
            delays = hstack((itd/2, -itd/2))
            fd = FractionalDelay(silence(1*ms, samplerate=samplerate), delays)
            ir = fd.impulse_response
            data = zeros((2, len(itd), fd.filter_length))
            data[0, :, :] = ir[:len(itd), :]
            data[1, :, :] = ir[len(itd):, :]
            self.delay_offset = fd.delay_offset
        self.hrtfset = HRTFSet(data, samplerate, coords)
        self.hrtfset.name = 'ITDDatabaseSubject'
        self.subjects = ['0']

    def load_subject(self, subject='0'):
        return self.hrtfset
