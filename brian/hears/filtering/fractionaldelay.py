from brian import *
from filterbank import *
from firfilterbank import *

__all__ = ['FractionalDelay']

class FractionalDelay(FIRFilterbank):
    '''
    Filterbank for applying delays which are fractional multiples of the timestep
    
    Initialised with arguments:
    
    ``source``
        Source sound or filterbank.
    ``delays``
        A list or array of delays to apply (the number of channels in the
        filterbank will be equal to the length of this).
    ``filter_length=None``
        Use this to explicitly set the length of the impulse response, should
        be odd. If not specified, it will be automatically determined from
        the delays. See notes below.
    ``**args``
        Arguments to pass to :class:`FIRFilterbank` (from which this class
        is derived).
        
    **Attributes**
    
    .. attribute:: delay_offset
    
        The global delay offset. If the specified delay in a given channel is
        ``delay`` the actual delay will be ``delay_offset+delay``. It is equal
        to ``(filter_length/2)/source.samplerate``.
        
    .. attribute:: filter_length
    
        The length of the filter to use. This is automatically determined
        from the delays. Note that ``delay_offset`` should be larger than the
        maximum positive or negative delay. The minimum filter length is
        by default 2048 samples, which allows for good accuracy for signals
        with power above 20 Hz. For low frequency analysis, longer filters will
        be necessary. For high frequency analysis, a shorter filter length could
        be used for a more efficient computation.
    
    **Notes**
    
    Inducing a delay for a sound that is an integer multiple of the timestep
    (1/samplerate) can be done simply by offsetting the samples, e.g.
    ``sound[3:]`` is ``sound`` delayed by ``3/sound.samplerate``. However,
    for fractional multiples of the timestep, the sound needs to be filtered.
    The theory and code for this was adapted from
    `http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html <http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html>`__.
    
    The filters induce a delay of ``delay_offset+delay`` where ``delay_offset``
    is a positive value larger than the maximum positive or negative delay.
    This value is available as the attribute ``delay_offset``.
    '''
    def __init__(self, source, delays, filter_length=None, **args):
        delays = array(delays)
        delay_max = amax(abs(delays))
        delay_max_int = int(ceil(source.samplerate*delay_max))
        if filter_length is None:
            filter_length = 2*int(delay_max_int*1.25)+1
            if filter_length<2048:
                filter_length = 2048
        if filter_length/2<=delay_max_int:
            raise ValueError('Filter length not long enough for selected delays.')
        self.delay_offset = (filter_length//2)/source.samplerate
        self.filter_length = filter_length
        self.delays = delays
        irs = [fractional_delay_ir(delay, source.samplerate,
                    filter_length=filter_length) for delay in delays]
        irs = array(irs)
        self.impulse_response = irs
        FIRFilterbank.__init__(self, source, irs, **args)

# Adapted from
# http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html    
def fractional_delay_ir(delay, samplerate, filter_length=151):
    delay = delay*samplerate
    centre_tap = filter_length // 2
    t = arange(filter_length)
    x = t-delay
    if abs(round(delay)-float(delay))<1e-10:
        return array(x==centre_tap, dtype=float)
    sinc = sin(pi*(x-centre_tap))/(pi*(x-centre_tap))
    window = 0.54-0.46*cos(2.0*pi*(x+0.5)/filter_length) # Hamming window
    tap_weight = window*sinc
    return tap_weight
