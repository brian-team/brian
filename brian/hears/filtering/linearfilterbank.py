from brian import *
from scipy import signal, weave, random
from filterbank import Filterbank, RestructureFilterbank
from ..bufferable import Bufferable
try:
    import numexpr
except ImportError:
    numexpr = None

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

if True:
    from itertools import izip
    alf_cache = {}
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
        if id(zi) in alf_cache:
            alf_cache_a, alf_cache_b, alf_cache_zi = alf_cache[id(zi)]
        else:
            alf_cache_b = [[0]*zi.shape[2] for _ in xrange(b.shape[1])]
            alf_cache_a = [[0]*zi.shape[2] for _ in xrange(b.shape[1])]
            alf_cache_zi = [[0]*zi.shape[2] for _ in xrange(b.shape[1])]
            for curf in xrange(zi.shape[2]):
                alf_cache_b[0][curf] = b[:, 0, curf]
                alf_cache_zi[0][curf] = zi[:, 0, curf]
                for i in xrange(b.shape[1]-2):
                    alf_cache_b[i+1][curf] = b[:, i+1, curf]
                    alf_cache_zi[i+1][curf] = zi[:, i+1, curf]
                    alf_cache_a[i+1][curf] = a[:, i+1, curf]
                i = b.shape[1]-2
                alf_cache_b[i+1][curf] = b[:, i+1, curf]
                alf_cache_a[i+1][curf] = a[:, i+1, curf]
        X = x
        output = empty_like(X)
        num_cascade = zi.shape[2]
        b_loop_size = b.shape[1]-2
        for sample, (x, o) in enumerate(izip(X, output)):
            for curf in xrange(num_cascade):
                #y = b[:, 0, curf]*x+zi[:, 0, curf]
                y = alf_cache_b[0][curf]*x
                add(y, alf_cache_zi[0][curf], y)
                for i in xrange(b_loop_size):
                    #zi[:, i, curf] = b[:, i+1, curf]*x+zi[:, i+1, curf]-a[:, i+1, curf]*y
                    t = alf_cache_b[i+1][curf]*x
                    add(t, alf_cache_zi[i+1][curf], t)
                    subtract(t, alf_cache_a[i+1][curf]*y, t)
                    alf_cache_zi[i][curf][:] = t
                i = b.shape[1]-2
                #zi[:, i, curf] = b[:, i+1, curf]*x-a[:, i+1, curf]*y
                t = alf_cache_b[i+1][curf]*x
                subtract(t, alf_cache_a[i+1][curf]*y, t)
                alf_cache_zi[i][curf][:] = t
                x = y
            #output[sample] = y
            o[:] = y
        return output

if numexpr is not None and False:
    def apply_linear_filterbank(b, a, x, zi):
        X = x
        output = empty_like(X)
        for sample in xrange(X.shape[0]):
            x = X[sample]
            for curf in xrange(zi.shape[2]):
                #y = b[:, 0, curf]*x+zi[:, 0, curf]
                y = numexpr.evaluate('b*x+zi', local_dict={
                        'b':b[:, 0, curf],
                        'x':x,
                        'zi':zi[:, 0, curf]})
                for i in xrange(b.shape[1]-2):
                    #zi[:, i, curf] = b[:, i+1, curf]*x+zi[:, i+1, curf]-a[:, i+1, curf]*y
                    zi[:, i, curf] = numexpr.evaluate('b*x+zi-a*y', local_dict={
                                            'b':b[:, i+1, curf],
                                            'x':x,
                                            'zi':zi[:, i+1, curf],
                                            'a':a[:, i+1, curf],
                                            'y':y})
                i = b.shape[1]-2
                #zi[:, i, curf] = b[:, i+1, curf]*x-a[:, i+1, curf]*y
                zi[:, i, curf] = numexpr.evaluate('b*x-a*y', local_dict={
                                        'b':b[:, i+1, curf],
                                        'x':x,
                                        'a':a[:, i+1, curf],
                                        'y':y})
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
    
    def apply_linear_filterbank(b, a, x, zi):
        if zi.shape[2]>1:
            # we need to do this so as not to alter the values in x in the C code below
            # but if zi.shape[2] is 1 there is only one filter in the chain and the
            # copy operation at the end of the C code will never happen.
            x = array(x, copy=True)
        y = empty_like(x)
        n, m, p = b.shape
        n1, m1, p1 = a.shape
        numsamples = x.shape[0]
        if n1!=n or m1!=m or p1!=p or x.shape!=(numsamples, n) or zi.shape!=(n, m-1, p):
            raise ValueError('Data has wrong shape.')
        if numsamples>1 and not x.flags['C_CONTIGUOUS']:
            raise ValueError('Input data must be C_CONTIGUOUS')
        if not b.flags['F_CONTIGUOUS'] or not a.flags['F_CONTIGUOUS'] or not zi.flags['F_CONTIGUOUS']:
            raise ValueError('Filter parameters must be F_CONTIGUOUS')
        code = '''
        #define X(s,i) x[(s)*n+(i)]
        #define Y(s,i) y[(s)*n+(i)]
        #define A(i,j,k) a[(i)+(j)*n+(k)*n*m]
        #define B(i,j,k) b[(i)+(j)*n+(k)*n*m]
        #define Zi(i,j,k) zi[(i)+(j)*n+(k)*n*(m-1)]
        for(int s=0; s<numsamples; s++)
        {
            for(int k=0; k<p; k++)
            {
                for(int j=0; j<n; j++)
                             Y(s,j) =   B(j,0,k)*X(s,j) + Zi(j,0,k);
                for(int i=0; i<m-2; i++)
                    for(int j=0;j<n;j++)
                        Zi(j,i,k) = B(j,i+1,k)*X(s,j) + Zi(j,i+1,k) - A(j,i+1,k)*Y(s,j);
                for(int j=0; j<n; j++)
                      Zi(j,m-2,k) = B(j,m-1,k)*X(s,j)               - A(j,m-1,k)*Y(s,j);
                if(k<p-1)
                    for(int j=0; j<n; j++)
                        X(s,j) = Y(s,j);
            }
        }
        '''
        if 0: # this version is used for comparing without vectorising over frequency
            code = '''
            #define X(s,i) x[(s)*n+(i)]
            #define Y(s,i) y[(s)*n+(i)]
            #define A(i,j,k) a[(i)+(j)*n+(k)*n*m]
            #define B(i,j,k) b[(i)+(j)*n+(k)*n*m]
            #define Zi(i,j,k) zi[(i)+(j)*n+(k)*n*(m-1)]
            for(int j=0; j<n; j++)
            {
                for(int k=0; k<p; k++)
                {
                    for(int s=0; s<numsamples; s++)
                    {
                    
                        Y(s,j) =   B(j,0,k)*X(s,j) + Zi(j,0,k);
                        for(int i=0; i<m-2; i++)
                            Zi(j,i,k) = B(j,i+1,k)*X(s,j) + Zi(j,i+1,k) - A(j,i+1,k)*Y(s,j);                    
                        Zi(j,m-2,k) = B(j,m-1,k)*X(s,j)               - A(j,m-1,k)*Y(s,j);
                        if(k<p-1)
                            X(s,j) = Y(s,j);
                    }
                }
            }
            '''
        weave.inline(code, ['b', 'a', 'x', 'zi', 'y', 'n', 'm', 'p', 'numsamples'],
                     compiler=_cpp_compiler,
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
        The coeffs b, a must be of shape ``(nchannels, m)`` or
        ``(nchannels, m, p)``. Here ``m`` is
        the order of the filters, and ``p`` is the number of filters in a
        chain (first you apply ``[:, :, 0]``, then ``[:, :, 1]``, etc.).
    
    The filter parameters are stored in the modifiable attributes ``filt_b``,
    ``filt_a`` and ``filt_state`` (the variable ``z`` in the section below).
    
    **Notes**
    
    These notes adapted from scipy's :func:`~scipy.signal.lfilter` function.
    
    The filterbank is implemented as a direct II transposed structure.
    This means that for a single channel and element of the filter cascade,
    the output y for an input x is defined by::

        a[0]*y[m] = b[0]*x[m] + b[1]*x[m-1] + ... + b[m]*x[0]
                              - a[1]*y[m-1] - ... - a[m]*y[0]

    using the following difference equations::

        y[i] = b[0]*x[i] + z[0,i-1]
        z[0,i] = b[1]*x[i] + z[1,i-1] - a[1]*y[i]
        ...
        z[m-3,i] = b[m-2]*x[i] + z[m-2,i-1] - a[m-2]*y[i]
        z[m-2,i] = b[m-1]*x[i] - a[m-1]*y[i]

    where i is the output sample number.

    The rational transfer function describing this filter in the
    z-transform domain is::
    
                                -1              -nb
                    b[0] + b[1]z  + ... + b[m] z
            Y(z) = --------------------------------- X(z)
                                -1              -na
                    a[0] + a[1]z  + ... + a[m] z
        
    '''
    def __init__(self, source, b, a):
        # Automatically duplicate mono input to fit the desired output shape
        if b.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, b.shape[0])
        Filterbank.__init__(self, source)
        # Weave version of filtering requires Fortran ordering of filter params
        if len(b.shape)==2 and len(a.shape)==2:
            b = reshape(b, b.shape+(1,))
            a = reshape(a, a.shape+(1,))
        self.filt_b = array(b, order='F')
        self.filt_a = array(a, order='F')
        self.filt_state = zeros((b.shape[0], b.shape[1]-1, b.shape[2]), order='F')

    def reset(self):
        self.buffer_init()
        
    def buffer_init(self):
        Filterbank.buffer_init(self)
        self.filt_state[:] = 0
    
    def buffer_apply(self, input):
        return apply_linear_filterbank(self.filt_b, self.filt_a, input,
                                       self.filt_state)

# Use the GPU version if available
try:
    if get_global_preference('brianhears_usegpu'):
        import pycuda
        from gpulinearfilterbank import LinearFilterbank
        use_gpu = True
    else:
        use_gpu = False
except ImportError:
    use_gpu = False
