'''
FIR filterbank, can be treated as a special case of LinearFilterbank, but an
optimisation is possible using buffered output by using FFT based convolution
as in HRTF.apply. To do this is slightly tricky because it needs to cache
previous inputs. For the moment, we implement it as a special case of
LinearFilterbank but later this will change to using the FFT method.
'''
from brian import *
from filterbank import *
from linearfilterbank import *

__all__ = ['FIRFilterbank']

class FIRFilterbank(LinearFilterbank):
    def __init__(self, source, impulseresponse):
        # if a 1D impulse response is given we apply it to every channel
        # Note that because we are using LinearFilterbank at the moment, this
        # means duplicating the impulse response. However, it could be stored
        # just once when we move to using FFT based convolution and in fact this
        # will save a lot of computation as the FFT only needs to be computed
        # once then.
        if len(impulseresponse.shape)==1:
            impulseresponse = repeat(reshape(impulseresponse, (1, len(impulseresponse))), source.nchannels, axis=0)
        # Automatically duplicate mono input to fit the desired output shape
        if impulseresponse.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, impulseresponse.shape[0])
        # For the moment, implement it as a LinearFilterbank
        b = reshape(impulseresponse, impulseresponse.shape+(1,))
        a = zeros_like(b)
        a[:, 0, :] = 1
        LinearFilterbank.__init__(self, source, b, a)
