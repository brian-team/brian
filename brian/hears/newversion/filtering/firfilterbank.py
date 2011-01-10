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

__all__ = ['FIRFilterbank', 'LinearFIRFilterbank', 'FFTFIRFilterbank']

class LinearFIRFilterbank(LinearFilterbank):
    def __init__(self, source, impulse_response, minimum_buffer_size=None):
        # if a 1D impulse response is given we apply it to every channel
        # Note that because we are using LinearFilterbank at the moment, this
        # means duplicating the impulse response. However, it could be stored
        # just once when we move to using FFT based convolution and in fact this
        # will save a lot of computation as the FFT only needs to be computed
        # once then.
        if len(impulse_response.shape)==1:
            impulse_response = repeat(reshape(impulse_response, (1, len(impulse_response))), source.nchannels, axis=0)
        # Automatically duplicate mono input to fit the desired output shape
        if impulse_response.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, impulse_response.shape[0])
        # Implement it as a LinearFilterbank
        b = reshape(impulse_response, impulse_response.shape+(1,))
        a = zeros_like(b)
        a[:, 0, :] = 1
        LinearFilterbank.__init__(self, source, b, a)
        if minimum_buffer_size is not None:
            self.minimum_buffer_size = minimum_buffer_size

class FFTFIRFilterbank(Filterbank):
    def __init__(self, source, impulse_response, minimum_buffer_size=None):
        # if a 1D impulse response is given we apply it to every channel
        # Note that because we are using LinearFilterbank at the moment, this
        # means duplicating the impulse response. However, it could be stored
        # just once when we move to using FFT based convolution and in fact this
        # will save a lot of computation as the FFT only needs to be computed
        # once then.
        if len(impulse_response.shape)==1:
            impulse_response = repeat(reshape(impulse_response, (1, len(impulse_response))), source.nchannels, axis=0)
        # Automatically duplicate mono input to fit the desired output shape
        if impulse_response.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, impulse_response.shape[0])
        Filterbank.__init__(self, source)

        self.input_cache = zeros((impulse_response.shape[1], self.nchannels))
        self.impulse_response = impulse_response
        self.fftcache_nmax = -1
        if minimum_buffer_size is None:
            minimum_buffer_size = 3*impulse_response.shape[1]
        self.minimum_buffer_size = minimum_buffer_size

    def buffer_init(self):
        Filterbank.buffer_init(self)
        self.input_cache[:] = 0

    # This version uses a single FFT/IFFT call, using the axis keyword, but it
    # doesn't appear to be any more efficient than looping, and uses much more
    # memory, although my tests weren't exhaustive.
#    def buffer_apply(self, input):
#        nmax = max(self.input_cache.shape[0]+input.shape[0], self.impulse_response.shape[1])
#        nmax = 2**int(ceil(log2(nmax)))
#        if self.fftcache_nmax!=nmax:
#            # impulse response: (nchannels, ir_length)
#            ir = zeros((self.nchannels, nmax))
#            ir[:, :self.impulse_response.shape[1]] = self.impulse_response
#            # fftcache: (ir_length, nchannels)
#            self.fftcache = fft(ir, n=nmax, axis=1).T
#            self.fftcache_nmax = nmax
#        fullinput = vstack((self.input_cache, input))
#        fullinput = vstack((fullinput, zeros((nmax-fullinput.shape[1], self.nchannels))))
#        # fullinput: (ir_length, nchannels)
#        fullinput_fft = fft(fullinput, n=nmax, axis=0)
#        fulloutput_fft = fullinput_fft*self.fftcache
#        fulloutput = ifft(fulloutput_fft, n=nmax, axis=0).real
#        output = fulloutput[self.input_cache.shape[0]:self.input_cache.shape[0]+input.shape[0]]
#        # update input cache
#        nic = self.input_cache.shape[0]
#        ni = input.shape[0]
#        #print ni, nic
#        if ni>=nic:
#            self.input_cache[:, :] = input[-nic:, :]
#        else:
#            self.input_cache[:-ni, :] = self.input_cache[ni:, :]
#            self.input_cache[-ni:, :] = input
#        return output
    
    def buffer_apply(self, input):
        output = zeros_like(input)
        nmax = max(self.input_cache.shape[0]+input.shape[0], self.impulse_response.shape[1])
        nmax = 2**int(ceil(log2(nmax)))
        if self.fftcache_nmax!=nmax:
            self.fftcache = []
        for i, (previnput, curinput, ir) in enumerate(zip(self.input_cache.T,
                                                          input.T,
                                                          self.impulse_response)):
            fullinput = hstack((previnput, curinput))
            # pad
            fullinput = hstack((fullinput, zeros(nmax-len(fullinput))))
            # apply fft
            if self.fftcache_nmax!=nmax:
                # recompute IR fft, first pad, then take fft, then store
                ir = hstack((ir, zeros(nmax-len(ir))))
                ir_fft = fft(ir, n=nmax)
                self.fftcache.append(ir_fft)
            else:
                ir_fft = self.fftcache[i]
            fullinput_fft = fft(fullinput, n=nmax)
            curoutput_fft = fullinput_fft*ir_fft
            curoutput = ifft(curoutput_fft)
            # unpad
            curoutput = curoutput[len(previnput):len(previnput)+len(curinput)]
            output[:, i] = curoutput.real
        if self.fftcache_nmax!=nmax:
            self.fftcache_nmax = nmax
        # update input cache
        nic = self.input_cache.shape[0]
        ni = input.shape[0]
        #print ni, nic
        if ni>=nic:
            self.input_cache[:, :] = input[-nic:, :]
        else:
            self.input_cache[:-ni, :] = self.input_cache[ni:, :]
            self.input_cache[-ni:, :] = input
        return output


class FIRFilterbank(Filterbank):
    '''
    Finite impulse response filterbank
    
    Initialisation parameters:
    
    ``source``
        Source sound or filterbank.
    ``impulse_response``
        Either a 1D array providing a single impulse response applied to every
        input channel, or a 2D array of shape ``(nchannels, ir_length)`` for
        ``ir_length`` the number of samples in the impulse response. Note that
        if you are using a multichannel sound ``x`` as a set of impulse responses,
        the array should be ``impulse_response=array(x.T)``.
    ``minimum_buffer_size=None``
        If specified, gives a minimum size to the buffer. By default, for the
        FFT convolution based implementation of ``FIRFilterbank``, the minimum
        buffer size will be ``3*ir_length``. For maximum efficiency with FFTs,
        ``buffer_size+ir_length`` should be a power of 2 (otherwise there will
        be some zero padding), and ``buffer_size`` should be as large as
        possible.
    '''
    def __init__(self, source, impulse_response, use_linearfilterbank=False,
                 minimum_buffer_size=None):
        if use_linearfilterbank:
            self.__class__ = LinearFIRFilterbank
        else:
            self.__class__ = FFTFIRFilterbank
        self.__init__(source, impulse_response,
                      minimum_buffer_size=minimum_buffer_size)
