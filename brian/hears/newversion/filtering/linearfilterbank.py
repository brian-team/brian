from brian import *
from scipy import signal, weave, random
from filterbank import Filterbank, RestructureFilterbank
from ..bufferable import Bufferable

# TODO: test all the buffered version of apply_linear_filterbank here
# So far, they seem to more or less work, but probably need more thorough
# testing.

__all__ = ['LinearFilterbank','apply_linear_filterbank']

def apply_linear_filterbank(b, a, x, zi):
    '''
    Parallel version of scipy lfilter command for a bank of n sequences of length 1
    
    In scipy.lfilter, you can apply a filter to multiple sounds at the same time,
    but you can't apply a bank of filters at the same time. This command does
    that. The coeffs b, a must be of shape (n,m,p), x must be of shape (s, n),
    and zi must be of shape (n,m-1,p). Here n is the number of channels in the
    filterbank, m is the order of the filter, p is the number of filters in
    a chain (cascade) to apply (you do first with (:,:,0) then (:,:,1), etc.),
    and s is the size of the buffer segment.
    '''
    X = x
    output = empty_like(X)
    for sample in xrange(X.shape[0]):
        x = X[sample]
        for curf in xrange(zi.shape[2]):
            y = b[:, 0, curf]*x+zi[:, 0, curf]
            for i in xrange(b.shape[1]-2):
                zi[:, i, curf] = b[:, i+1, curf]*x+zi[:, i+1, curf]-a[:, i+1, curf]*y
            i = b.shape[1]-2
            zi[:, i, curf] = b[:, i+1, curf]*x-a[:, i+1, curf]*y
            x = y
        output[sample] = y
    return output


# TODO: accelerate this even more using SWIG instead of weave?
if get_global_preference('useweave'):
    _cpp_compiler = get_global_preference('weavecompiler')
    _extra_compile_args = ['-O3']
    if _cpp_compiler=='gcc':
        _extra_compile_args += get_global_preference('gcc_options')
    _old_apply_linear_filterbank = apply_linear_filterbank
    
    # TODO: improve C code (very inefficient at the moment using blitz,
    # instead do it with pointers).
    def apply_linear_filterbank(b, a, x, zi):
        if zi.shape[2]>1:
            # we need to do this so as not to alter the values in x in the C code below
            # but if zi.shape[2] is 1 there is only one filter in the chain and the
            # copy operation at the end of the C code will never happen.
            x = array(x, copy=True)
#        if not isinstance(x, ndarray) or not len(x.shape) or x.shape==(1,):
#            newx=empty(b.shape[0])
#            newx[:]=x
#            x=newx
        y = empty_like(x)
        n, m, p = b.shape
        n1, m1, p1 = a.shape
        numsamples = x.shape[0]
        assert n1==n and m1==m and p1==p
        assert x.shape==(numsamples, n), str(x.shape)
        assert zi.shape==(n, m-1, p)
        code = '''
        for(int s=0; s<numsamples; s++)
        {
            for(int k=0; k<p; k++)
            {
                for(int j=0; j<n; j++)
                             y(s,j) =   b(j,0,k)*x(s,j) + zi(j,0,k);
                for(int i=0; i<m-2; i++)
                    for(int j=0;j<n;j++)
                        zi(j,i,k) = b(j,i+1,k)*x(s,j) + zi(j,i+1,k) - a(j,i+1,k)*y(s,j);
                for(int j=0; j<n; j++)
                      zi(j,m-2,k) = b(j,m-1,k)*x(s,j)               - a(j,m-1,k)*y(s,j);
                if(k<p-1)
                    for(int j=0; j<n; j++)
                        x(s,j) = y(s,j);
            }
        }
        '''
        weave.inline(code, ['b', 'a', 'x', 'zi', 'y', 'n', 'm', 'p', 'numsamples'],
                     compiler=_cpp_compiler,
                     type_converters=weave.converters.blitz,
                     extra_compile_args=_extra_compile_args)
        return y
    apply_linear_filterbank.__doc__ = _old_apply_linear_filterbank.__doc__


class LinearFilterbank(Filterbank):
    '''
    Generalised linear filterbank
    
    Initialisation arguments:

    ``source``
        The input to the filterbank, must have the same number of channels or
        just a single channel. In the latter case, the channels will be
        replicated.
    ``b``, ``a``
        The coeffs b, a must be of shape ``(nchannels, m, p)``. Here ``m`` is
        the order of the filters, and ``p`` is the number of filters in a
        chain (first you apply ``[:, :, 0]``, then ``[:, :, 1]``, etc.).    
    '''
    def __init__(self, source, b, a):
        # Automatically duplicate mono input to fit the desired output shape
        if b.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, b.shape[0])
        Filterbank.__init__(self, source)
        self.filt_b = b
        self.filt_a = a
        self.filt_state = zeros((b.shape[0], b.shape[1]-1, b.shape[2]))

    def reset(self):
        self.buffer_init()
        
    def buffer_init(self):
        Filterbank.buffer_init(self)
        self.filt_state[:] = 0
    
    def buffer_apply(self, input):
        return apply_linear_filterbank(self.filt_b, self.filt_a, input,
                                       self.filt_state)

# TODO: uncomment this when the GPU version is ready
# Use the GPU version if available
try:
    import pycuda
    from gpulinearfilterbank import LinearFilterbank
    use_gpu = True
except ImportError:
    use_gpu = False
