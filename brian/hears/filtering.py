'''
Various filters and filter interface

See docstrings for details. The faster gammatone filter is the GammatoneFB.
'''

from brian import *
from scipy import signal
from scipy import weave
try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    import pycuda.compiler
    from pycuda import gpuarray
    from brian.experimental.cuda.buffering import *
    import re
    def set_gpu_device(n):
        global _gpu_context
        autoinit.context.pop()
        _gpu_context = drv.Device(n).make_context()
    use_gpu = True
except ImportError:
    use_gpu = False
#use_gpu = False

__all__ = ['Filterbank', 'FilterbankChain', 'FilterbankGroup', 'FunctionFilterbank', 'ParallelLinearFilterbank',
           'parallel_lfilter_step',
           'HohmannGammatoneFilterbank', 'GammatoneFilterbank', 'MeddisGammatoneFilterbank',
           'IIRFilterbank', 'design_iir_filterbank', 'design_butterworth_filterbank', 'make_butterworth_filterbank',
           ]

def parallel_lfilter_step(b, a, x, zi):
    '''
    Parallel version of scipy lfilter command for a bank of n sequences of length 1
    
    In scipy.lfilter, you can apply a filter to multiple sounds at the same time,
    but you can't apply a bank of filters at the same time. This command does
    that. The coeffs b, a must be of shape (n,m,p), x must be of shape (n) or (),
    and zi must be of shape (n,m-1,p). Here n is the number of channels in the
    filterbank, m is the order of the filter, and p is the number of filters in
    a chain to apply (you do first with (:,:,0) then (:,:,1), etc.).
    '''
    for curf in xrange(zi.shape[2]):
        y           =        b[:,0,curf]*x + zi[:,0,curf]
        for i in xrange(b.shape[1]-2):
            zi[:,i,curf] = b[:,i+1,curf]*x + zi[:,i+1,curf] - a[:,i+1,curf]*y
        i = b.shape[1]-2
        zi[:,i,curf]     = b[:,i+1,curf]*x                  - a[:,i+1,curf]*y
        x = y
    return y

if get_global_preference('useweave'):
    # TODO: accelerate this even more using SWIG instead of weave
    _cpp_compiler = get_global_preference('weavecompiler')
    _old_parallel_lfilter_step = parallel_lfilter_step
    def parallel_lfilter_step(b, a, x, zi):
        if zi.shape[2]>1:
            # we need to do this so as not to alter the values in x in the C code below
            # but if zi.shape[2] is 1 there is only one filter in the chain and the
            # copy operation at the end of the C code will never happen.
            x = array(x, copy=True)
        if not isinstance(x, ndarray) or not len(x.shape) or x.shape==(1,):
            newx = empty(b.shape[0])
            newx[:] = x
            x = newx
        y = empty(b.shape[0])
        n, m, p = b.shape
        n1, m1, p1 = a.shape
        assert n1==n and m1==m and p1==p
        assert x.shape==(n,), str(x.shape)
        assert zi.shape==(n,m-1,p)
        code = '''
        for(int k=0;k<p;k++)
        {
            for(int j=0;j<n;j++)
                         y(j) =   b(j,0,k)*x(j) + zi(j,0,k);
            for(int i=0;i<m-2;i++)
                for(int j=0;j<n;j++)
                    zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
            for(int j=0;j<n;j++)
                  zi(j,m-2,k) = b(j,m-1,k)*x(j)               - a(j,m-1,k)*y(j);
            if(k<p-1)
                for(int j=0;j<n;j++)
                    x(j) = y(j);
        }
        '''
        weave.inline(code,['b', 'a', 'x', 'zi', 'y', 'n', 'm', 'p'],
                     compiler=_cpp_compiler,
                     type_converters=weave.converters.blitz,
                     extra_compile_args=['-O2'])
        return y
    parallel_lfilter_step.__doc__ = _old_parallel_lfilter_step.__doc__
       
def factorial(n):
    return prod(arange(1,n+1))

class Filterbank(object):
    '''
    Generalised filterbank object
    
    This class is a base class not designed to be instantiated. It defines the methods
    that a Filterbank object ought to have.
    
    Methods:
    
    ``__len__()``
        Returns the number of channels in the filter
    ``timestep(input)``
        Returns the filter output as an array of length ``len(filterbank)``. The ``input``
        should either be a scalar value or an array of scalar values of length
        ``len(filterbank)``. It represents the value of the sound at the current time
        step.
    ``apply(input)``
        Returns the filter applied to the time varying signal input. ``input`` can be
        1d in which case the value is the same at each point, or 2d in which case
        each row gives the values at a given time. Derived classes do not need to
        implement this.
    
    Properties:
    
    ``samplerate``
        The sample rate of the filterbank (in Hz).
    '''
    def timestep(self, input):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    samplerate = property(fget=lambda self:NotImplemented)
    
    def apply(self, input):
        return array([self.timestep(x) for x in input])

class FilterbankChain(Filterbank):
    '''
    Chains multiple filterbanks together
    
    Usage::
    
        FilterbankChain(filterbanks)
        
    Where ``filterbanks`` is a list of filters. The signal is fed into the first in
    this list, and then the output of that is fed into the next, and so on.
    '''
    def __init__(self, filterbanks):
        self.filterbanks = filterbanks
    
    def timestep(self, input):
        for fb in filterbanks:
            input = fb.timestep(input)
        return input
    
    def __len__(self):
        return len(self.filterbanks[0])
    
    samplerate = property(fget=lambda self:self.filterbanks[0].samplerate)

class FunctionFilterbank(Filterbank):
    '''
    Filterbank that just applies a given function
    '''
    def __init__(self, samplerate, N, func):
        self.fs = samplerate
        self.N = N
        self.func = func
    
    def timestep(self, input):
        return self.func(input)
    
    def __len__(self):
        return self.N
    
    samplerate = property(fget=lambda self:self.fs)   

class ParallelLinearFilterbank(Filterbank):
    '''
    Generalised parallel linear filterbank
    
    This filterbank allows you to construct a chain of linear filters in
    a bank so that each channel in the bank has its own filters. You pass
    the (b,a) parameters for the filter in the format specified in the
    function ``parallel_lfilter_step``.
    '''
    def __init__(self, b, a, samplerate=None):
        self.filt_b = b
        self.filt_a = a
        self.fs = samplerate
        self.N = b.shape[0]
        self.filt_state = zeros((b.shape[0], b.shape[1]-1, b.shape[2]))
    
    def reset(self):
        self.filt_state[:] = 0
    
    def __len__(self):
        return self.N
    
    samplerate = property(fget=lambda self:self.fs)   
    
    def timestep(self, input):
        if isinstance(input, ndarray):
            input = input.flatten()
        return parallel_lfilter_step(self.filt_b, self.filt_a, input, self.filt_state)

if use_gpu:
    
    nongpu_ParallelLinearFilterbank = ParallelLinearFilterbank
    
    class ParallelLinearFilterbank(Filterbank):
        '''
        Generalised parallel linear filterbank
        
        This filterbank allows you to construct a chain of linear filters in
        a bank so that each channel in the bank has its own filters. You pass
        the (b,a) parameters for the filter in the format specified in the
        function ``parallel_lfilter_step``.
        
        Note that there are additional GPU specific options:
        
        ``precision``
            Should be single for older GPUs.
        ``forcesync``
            Should be set to true to force a copy of GPU->CPU after each
            filter step, but not if the output is being used in ongoing
            GPU computations. By default this is ``True`` for compatibility.
        ``pagelocked_mem``
            Allocate faster pagelocked memory for GPU->CPU copies (doubles
            copy rate).
        ``unroll_filterorder``
            Whether or not to unroll the loop for the filter order, normally
            this should be done but not for very large filter orders.
        '''
        def __init__(self, b, a, samplerate=None,
                     precision='double', forcesync=True, pagelocked_mem=True, unroll_filterorder=True):
            if not use_gpu:
                self.__class__ = nongpu_ParallelLinearFilterbank
                self.__init__(b, a, samplerate=samplerate)
                return
            self.precision = precision
            if self.precision=='double':
                self.precision_dtype = float64
            else:
                self.precision_dtype = float32
            self.forcesync = forcesync
            self.pagelocked_mem = pagelocked_mem
            self.fs = samplerate
            self.N = b.shape[0]
            n, m, p = b.shape
            filt_b = array(b, dtype=self.precision_dtype)
            filt_a = array(a, dtype=self.precision_dtype)
            filt_state = zeros((n, m-1, p), dtype=self.precision_dtype)
            if pagelocked_mem:
                filt_y = drv.pagelocked_zeros((n,), dtype=self.precision_dtype)
                self.pre_x = drv.pagelocked_zeros((n,), dtype=self.precision_dtype)
            else:
                filt_y = zeros(n, dtype=self.precision_dtype)
                self.pre_x = zeros(n, dtype=self.precision_dtype)
            filt_x = zeros(n, dtype=self.precision_dtype)
            self.filt_b = gpuarray.to_gpu(filt_b.T.flatten()) # transform to Fortran order for better GPU mem
            self.filt_a = gpuarray.to_gpu(filt_a.T.flatten()) # access speeds
            self.filt_state = gpuarray.to_gpu(filt_state.T.flatten())
            self.filt_x = gpuarray.to_gpu(filt_x)
            self.filt_y = GPUBufferedArray(filt_y)
            code = '''
            #define x(i) _x[i]
            #define y(i) _y[i]
            #define a(i,j,k) _a[(i)+(j)*n+(k)*n*m]
            #define b(i,j,k) _b[(i)+(j)*n+(k)*n*m]
            #define zi(i,j,k) _zi[(i)+(j)*n+(k)*n*(m-1)]
            __global__ void filt(SCALAR *_b, SCALAR *_a, SCALAR *_x, SCALAR *_zi, SCALAR *_y)
            {
                int j = blockIdx.x * blockDim.x + threadIdx.x;
                if(j>=n) return;
            '''
            for k in range(p):
                loopcode = '''
                y(j) = b(j,0,k)*x(j) + zi(j,0,k);
                '''
                if unroll_filterorder:
                    for i in range(m-2):
                        loopcode += re.sub('\\bi\\b', str(i), '''
                        zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
                        ''')
                else:
                    loopcode += '''
                    for(int i=0;i<m-2;i++)
                        zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
                    '''
                loopcode += '''
                zi(j,m-2,k) = b(j,m-1,k)*x(j) - a(j,m-1,k)*y(j);
                '''
                if k<p-1:
                    loopcode += '''
                    x(j) = y(j);
                    '''
                loopcode = re.sub('\\bk\\b', str(k), loopcode)
                code += loopcode
            code += '''
            }
            '''
            code = code.replace('SCALAR', self.precision)
            code = re.sub("\\bp\\b", str(p), code)
            code = re.sub("\\bm\\b", str(m), code)
            code = re.sub("\\bn\\b", str(n), code)
            self.gpu_mod = pycuda.compiler.SourceModule(code)
            self.gpu_filt_func = self.gpu_mod.get_function("filt")
            blocksize = 512#self.maxblocksize
            if n<blocksize:
                blocksize = n
            if n%blocksize==0:
                gridsize = n/blocksize
            else:
                gridsize = n/blocksize+1
            self.block = (blocksize,1,1)
            self.grid = (gridsize,1)
            self.gpu_filt_func.prepare(('i','i','i','i','i'), self.block)
            self._has_run_once = False
    
        def reset(self):
            self.filt_state.set(zeros(self.filt_state.shape, dtype=self.filt_state.dtype))
        
        def __len__(self):
            return self.N
        
        samplerate = property(fget=lambda self:self.fs)   
        
        def timestep(self, input):
            b = self.filt_b
            a = self.filt_a
            x = input
            zi = self.filt_state
            y = self.filt_y
            fx = self.filt_x
            if isinstance(x, GPUBufferedArray):
                if not len(x.shape) or x.shape==(1,):
                    x.sync_to_cpu()
                    newx = empty(self.N, dtype=b.dtype)
                    newx[:] = x
                    fx.set(newx)
                else:
                    drv.memcpy_dtod(fx.gpu_dev_alloc, x.gpu_dev_alloc)
            else:
                if not isinstance(x, ndarray) or not len(x.shape) or x.shape==(1,):
                    # Current version of pycuda doesn't allow .fill(val) method on float64 gpuarrays
                    # because it assumed float32 only, so we have to do our own copying here
                    px = self.pre_x
                    px[:] = x
                    x = px
                fx.set(x)
            if self._has_run_once:
                self.gpu_filt_func.launch_grid(*self.grid)
            else:
                self.gpu_filt_func.prepared_call(self.grid, int(b.gpudata), int(a.gpudata), int(fx.gpudata),
                                                 int(zi.gpudata), y.gpu_pointer)
                self._has_run_once = True
            y.changed_gpu_data()
            if self.forcesync:
                y.sync_to_cpu()#might need to turn this on although it slows everything down
            return y
        
        def apply(self, input):
            return array([self.timestep(x).copy() for x in input])

class FilterbankGroupStateUpdater(StateUpdater):
    def __init__(self):
        pass
    
    def __call__(self, P):
        if P._x_stilliter is not None:
            try:
                P.input = P._x_iter.next()
            except StopIteration:
                P.input = 0
                P._x_stilliter = False
        P.output[:] = P.filterbank.timestep(P.input)

class FilterbankGroup(NeuronGroup):
    '''
    Allows a Filterbank object to be used as a NeuronGroup
    
    Initialised with variables:
    
    ``filterbank``
        The Filterbank object to be used by the group.
    ``x``
        The sound which the Filterbank will act on. If you don't specify
        this then you are in charge of updating the ``inp`` variable of
        the group each time step via a network operation.
    
    The variables of the group are:
    
    ``output``
        The output of the filterbank, multiple names are provided but they all
        do the same thing.
    ``input``
        The input to the filterbank.
    
    Has one additional method:
    
    .. method:: load_sound(x)
    
        Loads the sound 
    '''
    def __init__(self, filterbank, x=None):
        self.filterbank = filterbank
        fs = filterbank.samplerate
        eqs = '''
        output : 1
        input : 1
        '''
        NeuronGroup.__init__(self, len(filterbank), eqs, clock=Clock(dt=1/fs))
        self._state_updater = FilterbankGroupStateUpdater()
        fs = float(fs)
        self.load_sound(x)
        
    def load_sound(self, x):
        self._x = x
        if x is not None:
            self._x_iter = iter(self._x)
            self._x_stilliter = True
        else:
            self._x_iter = None
            self._x_stilliter = False
            
    def reinit(self):
        NeuronGroup.reinit(self)
        self.load_sound(self._x)

class HohmannGammatoneFilterbank(Filterbank):
    '''
    Approximate gammatone filter based on Hohmann paper
    
    Initialised with arguments:
    
    ``sampling_rate_hz``
        The sample rate in Hz.
    ``center_frequency_hz``
        A list or array of center frequencies.
    ``gamma_order``
        The order of the gammatone (4 by default).
    ``bandwidth_factor``
        How much larger the bandwidth parameter is compared to the ERB.
        
    The ERB values are computed according to the values in Hohmann 2002.
    '''
    GFB_L = 24.7  # see equation (17) in [Hohmann 2002]
    GFB_Q = 9.265 # see equation (17) in [Hohmann 2002]
    @check_units(sampling_rate_hz=Hz)
    def __init__(self, sampling_rate_hz, center_frequency_hz, gamma_order=4, bandwidth_factor=1.019):
        self.sampling_rate_hz = sampling_rate_hz
        sampling_rate_hz = float(sampling_rate_hz)
        center_frequency_hz = array(center_frequency_hz)
        self.gamma_order = gamma_order
        audiological_erb = (self.GFB_L + center_frequency_hz / self.GFB_Q) * bandwidth_factor
        # equation (14), line 3 [Hohmann 2002]:
        a_gamma          = (pi * factorial(2*self.gamma_order - 2) * \
                            2. ** -(2*self.gamma_order - 2) /        \
                            factorial(self.gamma_order - 1) ** 2)
        # equation (14), line 2 [Hohmann 2002]:
        b                = audiological_erb / a_gamma
        # equation (14), line 1 [Hohmann 2002]:
        lambda_          = exp(-2 * pi * b / sampling_rate_hz)
        # equation (10) [Hohmann 2002]:
        beta             = 2 * pi * center_frequency_hz / sampling_rate_hz
        # equation (1), line 2 [Hohmann 2002]:
        self.coefficient   = lambda_ * exp(1j * beta)
        self.beta = beta
        self.lambda_ = lambda_
        self.b = b
        self.a_gamma = a_gamma
        self.audiological_erb = audiological_erb
        self.center_frequency_hz = center_frequency_hz
        self.normalization_factor = \
            2 * (1 - abs(self.coefficient)) ** self.gamma_order
        self.state = zeros((len(center_frequency_hz), self.gamma_order))
        self.state = self.state*reshape(self.coefficient, self.coefficient.shape+(1,))
        
    def __len__(self):
        return len(self.center_frequency_hz)
    
    samplerate = property(fget=lambda self:self.sampling_rate_hz)
    
    def timestep(self, input):
        factor = self.normalization_factor
        filter_state = self.state
        for i in range(self.gamma_order):
            b = factor
            a = -self.coefficient
            z = filter_state[:,i]
            x = input
            bx = b*x
            y = bx+z
            z = bx-a*y
            input = y
            filter_state[:,i] = z
            factor = 1.0
        output = input
        return real(output)

class GammatoneFilterbank(ParallelLinearFilterbank):
    '''
    Exact gammatone based on Slaney's Auditory Toolbox for Matlab
    
    Initialised with arguments:
    
    ``fs``
        Sample rate (in Hz).
    ``fc``
        List or array of center frequencies.
    
    The ERBs are computed based on parameters in the Auditory Toolbox.
    '''
    # Change the following three parameters if you wish to use a different
    # ERB scale.  Must change in ERBSpace too.
    EarQ = 9.26449                #  Glasberg and Moore Parameters
    minBW = 24.7
    order = 1
    @check_units(fs=Hz)
    def __init__(self, fs, cf):
        cf = array(cf)
        self.cf = cf
        self.fs = fs
        fs = float(fs)
        EarQ, minBW, order = self.EarQ, self.minBW, self.order
        T = 1/fs
        ERB = ((cf/EarQ)**order + minBW**order)**(1/order)
        B = 1.019*2*pi*ERB
        self.B = B
        
        A0 = T
        A2 = 0
        B0 = 1
        B1 = -2*cos(2*cf*pi*T)/exp(B*T)
        B2 = exp(-2*B*T)
        
        A11 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3+2**1.5)*T*sin(2*cf*pi*T) / \
                exp(B*T))/2
        A12 = -(2*T*cos(2*cf*pi*T)/exp(B*T) - 2*sqrt(3+2**1.5)*T*sin(2*cf*pi*T) / \
                exp(B*T))/2
        A13 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3-2**1.5)*T*sin(2*cf*pi*T) / \
                exp(B*T))/2
        A14 = -(2*T*cos(2*cf*pi*T)/exp(B*T) - 2*sqrt(3-2**1.5)*T*sin(2*cf*pi*T) / \
                exp(B*T))/2
        
        i = 1j
        gain = abs((-2*exp(4*i*cf*pi*T)*T + \
                         2*exp(-(B*T) + 2*i*cf*pi*T)*T* \
                                 (cos(2*cf*pi*T) - sqrt(3 - 2**(3./2))* \
                                  sin(2*cf*pi*T))) * \
                   (-2*exp(4*i*cf*pi*T)*T + \
                     2*exp(-(B*T) + 2*i*cf*pi*T)*T* \
                      (cos(2*cf*pi*T) + sqrt(3 - 2**(3./2)) * \
                       sin(2*cf*pi*T)))* \
                   (-2*exp(4*i*cf*pi*T)*T + \
                     2*exp(-(B*T) + 2*i*cf*pi*T)*T* \
                      (cos(2*cf*pi*T) - \
                       sqrt(3 + 2**(3./2))*sin(2*cf*pi*T))) * \
                   (-2*exp(4*i*cf*pi*T)*T + 2*exp(-(B*T) + 2*i*cf*pi*T)*T* \
                   (cos(2*cf*pi*T) + sqrt(3 + 2**(3./2))*sin(2*cf*pi*T))) / \
                  (-2 / exp(2*B*T) - 2*exp(4*i*cf*pi*T) +  \
                   2*(1 + exp(4*i*cf*pi*T))/exp(B*T))**4)

        allfilts = ones(len(cf))
        
        self.A0, self.A11, self.A12, self.A13, self.A14, self.A2, self.B0, self.B1, self.B2, self.gain = \
            A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain
                   
        filt_a = dstack((array([ones(len(cf)), B1, B2]).T,)*4)
        filt_b = dstack((array([A0/gain, A11/gain, A2/gain]).T,
                         array([A0*ones(len(cf)), A12, zeros(len(cf))]).T,
                         array([A0*ones(len(cf)), A13, zeros(len(cf))]).T,
                         array([A0*ones(len(cf)), A14, zeros(len(cf))]).T))
        
        ParallelLinearFilterbank.__init__(self, filt_b, filt_a, fs*Hz)

class MeddisGammatoneFilterbank(ParallelLinearFilterbank):
    '''
    Parallel version of Ray Meddis' UTIL_gammatone.m
    '''
    # These consts from Hohmann gammatone code
    EarQ = 9.26449                #  Glasberg and Moore Parameters
    minBW = 24.7
    @check_units(fs=Hz)
    def __init__(self, fs, cf, order, bw):
        cf = array(cf)
        bw = array(bw)
        self.cf = cf
        self.fs = fs
        fs = float(fs)
        dt = 1/fs
        phi = 2 * pi * bw * dt
        theta = 2 * pi * cf * dt
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        alpha = -exp(-phi) * cos_theta
        b0 = ones(len(cf))
        b1 = 2 * alpha
        b2 = exp(-2 * phi)
        z1 = (1 + alpha * cos_theta) - (alpha * sin_theta) * 1j
        z2 = (1 + b1 * cos_theta) - (b1 * sin_theta) * 1j
        z3 = (b2 * cos(2 * theta)) - (b2 * sin(2 * theta)) * 1j
        tf = (z2 + z3) / z1
        a0 = abs(tf)
        a1 = alpha * a0   
        # we apply the same filters order times so we just duplicate them in the 3rd axis for the parallel_lfilter_step command
        a = dstack((array([b0, b1, b2]).T,)*order)
        b = dstack((array([a0, a1, zeros(len(cf))]).T,)*order)
        self.order = order
        
        ParallelLinearFilterbank.__init__(self, b, a, fs*Hz)    

class IIRFilterbank(Filterbank):
    @check_units(samplerate=Hz)
    def __init__(self, samplerate, N, b, a):
        self.fs = samplerate
        self.a = a
        self.b = b
        self.N = N
        self.zi = zeros((N, max(len(a), len(b))-1))
    
    def timestep(self, input):
        input = reshape(input, (self.N,1))
        y, self.zi = signal.lfilter(self.b, self.a, input, zi=self.zi)
        return y
    
    def apply_single(self, input):
        return signal.lfilter(self.b, self.a, input)
    
    def __len__(self):
        return self.N
    
    samplerate = property(fget=lambda self:self.fs)

@check_units(samplerate=Hz)
def design_iir_filterbank(samplerate, N, passband, stopband, gpass, gstop, ftype):
    '''
    Filterbank using scipy.signal.iirdesign
    
    Arguments:
    
    ``samplerate``
        The sample rate in Hz.
    ``N``
        The number of channels in the bank
    ``passband``, ``stopband``
        The edges of the pass and stop bands in Hz. For a lowpass filter, make
        passband<stopband and for a highpass make stopband>passband. For a
        bandpass or bandstop filter, make passband and stopband a list with
        two elements, e.g. for a bandpass have passband=[200*Hz, 500*hz] and
        stopband=[100*Hz, 600*Hz], or for a bandstop switch passband and stopband.
    ``gpass``
        The maximum loss in the passband in dB.
    ``gstop``
        The minimum attenuation in the stopband in dB.
    ``ftype``
        The type of IIR filter to design:
            elliptic    : 'ellip'
            Butterworth : 'butter',
            Chebyshev I : 'cheby1',
            Chebyshev II: 'cheby2',
            Bessel :      'bessel'
    
    See the documentation for scipy.signal.iirdesign for more details.
    '''
    # passband can take form x or (a,b) in Hz and we need to convert to scipy's format
    try:
        try:
            a, b = passband
            a = a/samplerate
            b = b/samplerate
            passband = [a,b]
            a+1 
            b+1
        except TypeError:
            passband = passband/samplerate
            passband+1
        try:
            a, b = stopband
            a = a/samplerate
            b = b/samplerate
            stopband = [a,b]
            a+1 
            b+1
        except TypeError:
            stopband = stopband/samplerate
            stopband+1
    except DimensionMismatchError:
        raise DimensionMismatchError('IIRFilterbank passband, stopband parameters must be in Hz')
    # now design filterbank
    b, a = signal.iirdesign(passband, stopband, gpass, gstop, ftype=ftype)
    fb = IIRFilterbank(samplerate, N, b, a)
    fb.N = N
    fb.passband = passband
    fb.stopband = stopband
    fb.gpass = gpass
    fb.gstop = gstop
    fb.ftype= ftype
    return fb

def design_butterworth_filterbank(samplerate, N, passband, stopband, gpass, gstop):
    '''
    Design a butterworth filterbank
    
    See docs for design_iir_filterbank for details.
    '''
    return design_iir_filterbank(samplerate, N, passband, stopband, gpass, gstop, ftype='butter')

def make_butterworth_filterbank(samplerate, N, ord, Wn, btype='low'):
    '''
    Make a butterworth filterbank directly
    
    Alternatively, use design_butterworth_filterbank
    
    Parameters:
    
    ``samplerate``
        Sample rate.
    ``N``
        Number of filters in the bank.
    ``ord``
        Order of the filter.
    ``Wn``
        Cutoff parameter(s) in Hz, either a single value or pair for band filters.
    ``btype``
        One of 'lowpass', 'highpass', 'bandpass' or 'bandstop'.
    '''
    try:
        try:
            Wn1, Wn2 = Wn
            Wn = (Wn1/samplerate+0.0, Wn2/samplerate+0.0)
        except TypeError:
            Wn = Wn/samplerate+0.0
    except DimensionMismatchError:
        raise DimensionMismatchError('Wn must be in Hz')
    b, a = signal.butter(ord, Wn, btype=btype)
    return IIRFilterbank(samplerate, N, b, a)

if __name__=='__main__':
    from sounds import *
    from erb import *
    sound = whitenoise(100*ms).ramp()
    defaultclock.dt = 1/sound.rate 
    fb = GammatoneFilterbank(sound.rate, erbspace(100*Hz, 5*kHz, 4))
    G = FilterbankGroup(fb, sound)
    neurons=NeuronGroup(len(G),model='''dv/dt=(I-v)/(5*ms):1
                                        I:1''', reset=0, threshold=1)
    neurons.I = linked_var(G, 'output', func=lambda x:50*clip(x,0,Inf))
    fb_mon = StateMonitor(G, 'output', record=True)
    neuron_mon = StateMonitor(neurons, 'v', record=True)
    spikes = SpikeMonitor(neurons)
    run(sound.duration)
    G.load_sound(tone(500*Hz, 100*ms).ramp())
    run(100*ms)
    subplot(311)
    fb_mon.plot()
    subplot(312)
    neuron_mon.plot()
    subplot(313)
    raster_plot(spikes)
    show()