'''
Various filters and filter interface

See docstrings for details. The faster gammatone filter is the GammatoneFB.
'''

from brian import *
from gputools import *
from scipy import signal
from scipy import weave
from scipy import random
from sounds import OnlineSound
from numpy import intp
# NOTE TO BERTRAND:
# DO NOT COMMENT THESE LINES OUT!-> Ok :)
try:
    import pycuda
    #import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    import pycuda.compiler
    from pycuda import gpuarray
    from brian.experimental.cuda.buffering import *
    import re
    #set_gpu_device(0)
#    def set_gpu_device(n):
#        global _gpu_context
#        autoinit.context.pop()
#        _gpu_context = drv.Device(n).make_context()
    use_gpu = True
except ImportError:
    use_gpu = False
#use_gpu=False

__all__=['GammachirpFilterbankFIR', 'GammachirpFilterbankIIR', 'Filterbank', 'FilterbankChain', 'FilterbankGroup', 'FunctionFilterbank', 'ParallelLinearFilterbank',
           'parallel_lfilter_step', 'GammatoneFilterbank',
           'IIRFilterbank' ,'MixFilterbank','TimeVaryingIIRFilterbank','MeddisGammatoneFilterbank','ButterworthFilterbank','CascadeFilterbank']


def parallel_lfilter_step(b, a, x, zi):
    '''
    Parallel version of scipy lfilter command for a bank of n sequences of length 1
    
    In scipy.lfilter, you can apply a filter to multiple sounds at the same time,
    but you can't apply a bank of filters at the same time. This command does
    that. The coeffs b, a must be of shape (n,m,p), x must be of shape (n) or (),
    and zi must be of shape (n,m-1,p). Here n is the number of channels in the
    filterbank, m is the order of the filter, and p is the number of filters in
    a chain (cascade) to apply (you do first with (:,:,0) then (:,:,1), etc.).
    '''
    for curf in xrange(zi.shape[2]):
        y=b[:, 0, curf]*x+zi[:, 0, curf]
        for i in xrange(b.shape[1]-2):
            zi[:, i, curf]=b[:, i+1, curf]*x+zi[:, i+1, curf]-a[:, i+1, curf]*y
        i=b.shape[1]-2
        zi[:, i, curf]=b[:, i+1, curf]*x-a[:, i+1, curf]*y
        x=y
    return y

if get_global_preference('useweave'):
    # TODO: accelerate this even more using SWIG instead of weave
    _cpp_compiler=get_global_preference('weavecompiler')
    _extra_compile_args=['-O3']
    if _cpp_compiler=='gcc':
        _extra_compile_args+=get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
    _old_parallel_lfilter_step=parallel_lfilter_step
    
    def parallel_lfilter_step(b, a, x, zi):
        if zi.shape[2]>1:
            # we need to do this so as not to alter the values in x in the C code below
            # but if zi.shape[2] is 1 there is only one filter in the chain and the
            # copy operation at the end of the C code will never happen.
            x=array(x, copy=True)
        if not isinstance(x, ndarray) or not len(x.shape) or x.shape==(1,):
            newx=empty(b.shape[0])
            newx[:]=x
            x=newx
        y=empty(b.shape[0])
        n, m, p=b.shape
        n1, m1, p1=a.shape
        assert n1==n and m1==m and p1==p
        assert x.shape==(n,), str(x.shape)
        assert zi.shape==(n, m-1, p)
        code='''
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
        weave.inline(code, ['b', 'a', 'x', 'zi', 'y', 'n', 'm', 'p'],
                     compiler=_cpp_compiler,
                     type_converters=weave.converters.blitz,
                     extra_compile_args=_extra_compile_args)
        return y
    parallel_lfilter_step.__doc__=_old_parallel_lfilter_step.__doc__

def factorial(n):
    return prod(arange(1, n+1))

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
    def __add__ (fb1,fb2):
        #print fb1,fb2
        return MixFilterbank(fb1,fb2)
        
    def __sub__ (fb1,fb2):
        return MixFilterbank(fb1,fb2,array([1,-1]))  
        
    def __rmul__(fb,scalar):
        func=lambda x:scalar*x
        return FilterbankChain([fb,FunctionFilterbank(fb.fs, fb.N, func)])
           
        
    def timestep(self, input):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    samplerate=property(fget=lambda self:NotImplemented)

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
        self.filterbanks=filterbanks
        self.fs=filterbanks[0].fs
        self.N=filterbanks[0].N
        
    def timestep(self, input):
        for fb in self.filterbanks:
            input=fb.timestep(input)
        return input

    def __len__(self):
        return len(self.filterbanks[0])

    samplerate=property(fget=lambda self:self.filterbanks[0].samplerate)

class FunctionFilterbank(Filterbank):
    '''
    Filterbank that just applies a given function
    '''
    def __init__(self, samplerate, N, func):
        self.fs=samplerate
        self.N=N
        self.func=func

    def timestep(self, input):
        # self.output[:]=self.func(input)
        return self.func(input)

    def __len__(self):
        return self.N

    samplerate=property(fget=lambda self:self.fs)

class ParallelLinearFilterbank(Filterbank):
    '''
    Generalised parallel linear filterbank
    
    This filterbank allows you to construct a chain of linear filters in
    a bank so that each channel in the bank has its own filters. You pass
    the (b,a) parameters for the filter in the format specified in the
    function ``parallel_lfilter_step``.
    '''
    def __init__(self, b, a, samplerate=None):
        self.filt_b=b
        self.filt_a=a
        self.fs=samplerate
        self.N=b.shape[0]
        self.filt_state=zeros((b.shape[0], b.shape[1]-1, b.shape[2]))

    def reset(self):
        self.filt_state[:]=0

    def __len__(self):
        return self.N

    samplerate=property(fget=lambda self:self.fs)

    def timestep(self, input):
        if isinstance(input, ndarray):
            input=input.flatten()
        return parallel_lfilter_step(self.filt_b, self.filt_a, input, self.filt_state)

if use_gpu:

    nongpu_ParallelLinearFilterbank=ParallelLinearFilterbank

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
                self.__class__=nongpu_ParallelLinearFilterbank
                self.__init__(b, a, samplerate=samplerate)
                return
            if pycuda.context is None:
                set_gpu_device(0)
            self.precision=precision
            if self.precision=='double':
                self.precision_dtype=float64
            else:
                self.precision_dtype=float32
            self.forcesync=forcesync
            self.pagelocked_mem=pagelocked_mem
            self.fs=samplerate
            self.N=b.shape[0]
            n, m, p=b.shape
            self.filt_b=b
            self.filt_a=a
            filt_b_gpu=array(b, dtype=self.precision_dtype)
            filt_a_gpu=array(a, dtype=self.precision_dtype)
            filt_state=zeros((n, m-1, p), dtype=self.precision_dtype)
            if pagelocked_mem:
                filt_y=drv.pagelocked_zeros((n,), dtype=self.precision_dtype)
                self.pre_x=drv.pagelocked_zeros((n,), dtype=self.precision_dtype)
            else:
                filt_y=zeros(n, dtype=self.precision_dtype)
                self.pre_x=zeros(n, dtype=self.precision_dtype)
            filt_x=zeros(n, dtype=self.precision_dtype)
            self.filt_b_gpu=gpuarray.to_gpu(filt_b_gpu.T.flatten()) # transform to Fortran order for better GPU mem
            self.filt_a_gpu=gpuarray.to_gpu(filt_a_gpu.T.flatten()) # access speeds
            self.filt_state=gpuarray.to_gpu(filt_state.T.flatten())
            self.filt_x=gpuarray.to_gpu(filt_x)
            self.filt_y=GPUBufferedArray(filt_y)
            #self.filt_y=gpuarray.to_gpu(filt_y)
            code='''
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
                loopcode='''
                y(j) = b(j,0,k)*x(j) + zi(j,0,k);
                '''
                if unroll_filterorder:
                    for i in range(m-2):
                        loopcode+=re.sub('\\bi\\b', str(i), '''
                        zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
                        ''')
                else:
                    loopcode+='''
                    for(int i=0;i<m-2;i++)
                        zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
                    '''
                loopcode+='''
                zi(j,m-2,k) = b(j,m-1,k)*x(j) - a(j,m-1,k)*y(j);
                '''
                if k<p-1:
                    loopcode+='''
                    x(j) = y(j);
                    '''
                loopcode=re.sub('\\bk\\b', str(k), loopcode)
                code+=loopcode
            code+='''
            }
            '''
            code=code.replace('SCALAR', self.precision)
            code=re.sub("\\bp\\b", str(p), code) #replace the variable by their values
            code=re.sub("\\bm\\b", str(m), code)
            code=re.sub("\\bn\\b", str(n), code)
            #print code
            self.gpu_mod=pycuda.compiler.SourceModule(code)
            self.gpu_filt_func=self.gpu_mod.get_function("filt")
            blocksize=512#self.maxblocksize
            if n<blocksize:
                blocksize=n
            if n%blocksize==0:
                gridsize=n/blocksize
            else:
                gridsize=n/blocksize+1
            self.block=(blocksize, 1, 1)
            self.grid=(gridsize, 1)
            self.gpu_filt_func.prepare((intp, intp, intp, intp, intp), self.block)
            self._has_run_once=False

        def reset(self):
            self.filt_state.set(zeros(self.filt_state.shape, dtype=self.filt_state.dtype))

        def __len__(self):
            return self.N

        samplerate=property(fget=lambda self:self.fs)

        def timestep(self, input):
            b=self.filt_b_gpu
            a=self.filt_a_gpu
            x=input
            zi=self.filt_state
            y=self.filt_y
            fx=self.filt_x
            if isinstance(x, GPUBufferedArray):
                if not len(x.shape) or x.shape==(1,):
                    x.sync_to_cpu()
                    newx=empty(self.N, dtype=b.dtype)
                    newx[:]=x
                    fx.set(newx)
                else:
                    drv.memcpy_dtod(fx.gpudata, x.gpu_dev_alloc, fx.size*self.precision_dtype().nbytes)
            else:
                if not isinstance(x, ndarray) or not len(x.shape) or x.shape==(1,):
                    # Current version of pycuda doesn't allow .fill(val) method on float64 gpuarrays
                    # because it assumed float32 only, so we have to do our own copying here
                    px=self.pre_x
                    px[:]=x
                    x=px
                fx.set(x)
            if self._has_run_once:
                self.gpu_filt_func.launch_grid(*self.grid)
            else:
                self.gpu_filt_func.prepared_call(self.grid, intp(b.gpudata), intp(a.gpudata), intp(fx.gpudata),
                                                 intp(zi.gpudata), y.gpu_pointer)
                self._has_run_once=True
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
                P.input=P._x_iter.next()
            except StopIteration:
                P.input=0
                P._x_stilliter=False
        else:
            P.input=P._x.update()
        P.output[:]=P.filterbank.timestep(P.input)

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
        self.filterbank=filterbank
        fs=filterbank.samplerate
        eqs='''
        output : 1
        input : 1
        '''
        NeuronGroup.__init__(self, len(filterbank), eqs, clock=Clock(dt=1/fs))
        self._state_updater=FilterbankGroupStateUpdater()
        fs=float(fs)
        self.load_sound(x)

    def load_sound(self, x):
        self._x=x
        
        if isinstance(x,OnlineSound):
            self._x_iter=None
            self._x_stilliter=None

        elif x is not None:
            self._x_iter=iter(self._x)
            self._x_stilliter=True
        else:
            self._x_iter=None
            self._x_stilliter=False

        
    def reinit(self):
        NeuronGroup.reinit(self)
        self.load_sound(self._x)

class MixFilterbank(Filterbank):
    '''  
    Mix filterbanks together with a given weight vectors
    '''
    
    def __init__(self, *args, **kwargs):#weights=None,

        if kwargs.get('weights')==None:
            weights=ones((len(args)))
        else:
            weights=kwargs.get('weights')
        self.filterbanks=args
        #print  args
        self.fs=args[0].fs
        self.nbr_fb=len(self.filterbanks)
        self.N=args[0].N
        self.output=zeros(self.N)
        self.weights=weights

    def timestep(self, input):
        self.output=zeros(self.N)
        for ind, fb in zip(range((self.nbr_fb)), self.filterbanks):
            self.output+=self.weights[ind]*fb.timestep(input)
        return self.output

    def __len__(self):
        return self.N
    samplerate=property(fget=lambda self:self.fs)

class CascadeFilterbank(ParallelLinearFilterbank):
    '''
    Cascade of a filterbank (nbr_cascade times)
    '''
    
    def __init__(self, filterbank,nbr_cascade):
        b=filterbank.filt_b
        a=filterbank.filt_a
        self.fs=filterbank.fs
        self.N=filterbank.N
        self.filt_b=zeros((b.shape[0], b.shape[1],nbr_cascade))
        self.filt_a=zeros((a.shape[0], a.shape[1],nbr_cascade))
        for i in range((nbr_cascade)):
            self.filt_b[:,:,i]=b[:,:,0]
            self.filt_a[:,:,i]=a[:,:,0]
            
        ParallelLinearFilterbank.__init__(self, self.filt_b, self.filt_a, self.fs*Hz)

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
    EarQ=9.26449                #  Glasberg and Moore Parameters
    minBW=24.7
    order=1
    
    @check_units(fs=Hz)
    def __init__(self, fs, cf):

        cf = array(cf)
        self.cf = cf
        self.N = len(cf)
        self.fs = fs
        fs = float(fs)
        EarQ, minBW, order = self.EarQ, self.minBW, self.order
        T = 1/fs
        ERB = ((cf/EarQ)**order + minBW**order)**(1/order)
        B = 1.019*2*pi*ERB
        self.B = B
        self.order=order
        A0 = T
        A2 = 0
        B0 = 1
        B1 = -2*cos(2*cf*pi*T)/exp(B*T)
        B2 = exp(-2*B*T)
        
        A11 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3+2**1.5)*T*sin(2*cf*pi*T) / \

                exp(B*T))/2
        A12=-(2*T*cos(2*cf*pi*T)/exp(B*T)-2*sqrt(3+2**1.5)*T*sin(2*cf*pi*T)/\
                exp(B*T))/2
        A13=-(2*T*cos(2*cf*pi*T)/exp(B*T)+2*sqrt(3-2**1.5)*T*sin(2*cf*pi*T)/\
                exp(B*T))/2
        A14=-(2*T*cos(2*cf*pi*T)/exp(B*T)-2*sqrt(3-2**1.5)*T*sin(2*cf*pi*T)/\
                exp(B*T))/2

        i=1j
        gain=abs((-2*exp(4*i*cf*pi*T)*T+\
                         2*exp(-(B*T)+2*i*cf*pi*T)*T*\
                                 (cos(2*cf*pi*T)-sqrt(3-2**(3./2))*\
                                  sin(2*cf*pi*T)))*\
                   (-2*exp(4*i*cf*pi*T)*T+\
                     2*exp(-(B*T)+2*i*cf*pi*T)*T*\
                      (cos(2*cf*pi*T)+sqrt(3-2**(3./2))*\
                       sin(2*cf*pi*T)))*\
                   (-2*exp(4*i*cf*pi*T)*T+\
                     2*exp(-(B*T)+2*i*cf*pi*T)*T*\
                      (cos(2*cf*pi*T)-\
                       sqrt(3+2**(3./2))*sin(2*cf*pi*T)))*\
                   (-2*exp(4*i*cf*pi*T)*T+2*exp(-(B*T)+2*i*cf*pi*T)*T*\
                   (cos(2*cf*pi*T)+sqrt(3+2**(3./2))*sin(2*cf*pi*T)))/\
                  (-2/exp(2*B*T)-2*exp(4*i*cf*pi*T)+\
                   2*(1+exp(4*i*cf*pi*T))/exp(B*T))**4)

        allfilts=ones(len(cf))

        self.A0, self.A11, self.A12, self.A13, self.A14, self.A2, self.B0, self.B1, self.B2, self.gain=\
            A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain

        filt_a=dstack((array([ones(len(cf)), B1, B2]).T,)*4)
        filt_b=dstack((array([A0/gain, A11/gain, A2/gain]).T,
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
        self.N = len(cf)
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
            
class GammachirpFilterbankIIR(ParallelLinearFilterbank):
    '''
    Implementaion of the gammachirp filter with logarithmic chirp as a cascade of a 4 second order IIR gammatone filter 
    and a 4 second orders asymmetric compensation filters
    From Unoki et al. 2001, Improvement of an IIR asymmetric compensation gammachirp filter,  
     
     comment: no GPU implementation so far... because
     c determines the rate of the frequency modulation or the chirp rate
     center_frequency 
     fr is the center frequency of the gamma tone (note: it is note the peak frequency of the gammachirp)
     '''
     
     
    def __init__(self, samplerate, fr, c=None):
        fr = array(fr)

        self.fr = fr
        self.N = len(fr)
        self.fs= samplerate


        if c==None:
            c=1*ones((fr.shape))
            
        self.c=c
        gammatone=GammatoneFilterbank(samplerate, fr)
        samplerate=float(samplerate)
        order=gammatone.order

        self.gammatone_filt_b=gammatone.filt_b
        self.gammatone_filt_a=gammatone.filt_a

        ERBw=24.7*(4.37e-3*fr+1.)
        compensation_filter_order=4
        b=1.019*ones((fr.shape))

        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.asymmetric_filt_b=zeros((len(fr), 2*order+1, 4))
        self.asymmetric_filt_a=zeros((len(fr), 2*order+1, 4))

        for k in arange(compensation_filter_order):

            r=exp(-p1*(p0/p4)**(k)*2*pi*b*ERBw/samplerate) #k instead of k-1 because range 0 N-1

            Dfr=(p0*p4)**(k)*p2*c*b*ERBw

            phi=2*pi*maximum((fr+Dfr), 0)/samplerate
            psy=2*pi*maximum((fr-Dfr), 0)/samplerate

            ap=vstack((ones(r.shape),-2*r*cos(phi), r**2)).T
            bz=vstack((ones(r.shape),-2*r*cos(psy), r**2)).T

            fn=fr+ compensation_filter_order* p3 *c *b *ERBw/4;

            vwr=exp(1j*2*pi*fn/samplerate)
            vwrs=vstack((ones(vwr.shape), vwr, vwr**2)).T

            ##normilization stuff
            nrm=abs(sum(vwrs*ap, 1)/sum(vwrs*bz, 1))
            temp=ones((bz.shape))
            for i in range((len(nrm))):
                temp[i, :]=nrm[i]
            bz=bz*temp

            self.asymmetric_filt_b[:, :, k]=bz
            self.asymmetric_filt_a[:, :, k]=ap
        #print B.shape,A.shape,Btemp.shape,Atemp.shape    
        #concatenate the gammatone filter coefficients so that everything is in cascade in each frequency channel
        #print self.gammatone_filt_b,self.asymmetric_filt_b
        self.filt_b=concatenate([self.gammatone_filt_b, self.asymmetric_filt_b],axis=2)
        self.filt_a=concatenate([self.gammatone_filt_a, self.asymmetric_filt_a],axis=2)

        ParallelLinearFilterbank.__init__(self, self.filt_b, self.filt_a, samplerate*Hz)

class GammachirpFilterbankFIR(ParallelLinearFilterbank):
    '''
    Fit of a auditory filter (from a reverse correlation) at the NM of a barn owl at 4.6 kHz. The tap of the FIR filter
    are the time response of the filter which is long. It is thus very slow
    
    '''
    def __init__(self, fs, fr,c=None,time_constant=None):
        fr = array(fr)
        self.fr = fr
        self.N = len(fr)
        self.fs = fs

        #%x = [amplitude, delay, time constant, frequency, phase, bias, IF glide slope]
        if c==None:
            x=array([0.8932, 0.7905 , 0.3436  , 4.6861  ,-4.4308 ,-0.0010  , 0.3453])

        if time_constant==None:
            x=array([0.8932, 0.7905 , 0.3436  , 4.6861  ,-4.4308 ,-0.0010  , 0.3453])
        x[-1]=c
        fs=float(fs)
        t=arange(0, 4, 1./fs*1000)
        LenGC=len(t)
        filt_b=zeros((1, LenGC, 1))
        filt_a=zeros((1, LenGC, 1))

        g=4
        tmax=x[2]*(g-1)
        G=x[0]/(tmax**(g-1)*exp(1-g))*(t-x[1]+tmax)**(g-1)*exp(-(t-x[1]+tmax)/x[2])*cos(2*pi*(x[3]*(t-x[1])+x[6]/2*(t-x[1])**2)+x[4])+x[5]

        filt_b[0, :, 0]=G
        filt_a[0, 0, 0]=1

        ParallelLinearFilterbank.__init__(self, filt_b, filt_a, fs*Hz)


class IIRFilterbank(ParallelLinearFilterbank):


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
    
    def __init__(self, samplerate, N, passband, stopband, gpass, gstop, ftype):
        # passband can take form x or (a,b) in Hz and we need to convert to scipy's format
        try:
            try:
                a, b=passband
                a=a/samplerate
                b=b/samplerate
                passband=[a, b]
                a+1
                b+1
            except TypeError:
                passband=passband/samplerate
                passband+1
            try:
                a, b=stopband
                a=a/samplerate
                b=b/samplerate
                stopband=[a, b]
                a+1
                b+1
            except TypeError:
                stopband=stopband/samplerate
                stopband+1
        except DimensionMismatchError:
            raise DimensionMismatchError('IIRFilterbank passband, stopband parameters must be in Hz')

        # now design filterbank

        self.fs=samplerate
        self.filt_b, self.filt_a = signal.iirdesign(passband, stopband, gpass, gstop, ftype=ftype)
        self.filt_b=kron(ones((N,1)),self.filt_b)
        self.filt_b=self.filt_b.reshape(self.filt_b.shape[0],self.filt_b.shape[1],1)
        self.filt_a=kron(ones((N,1)),self.filt_a)
        self.filt_a=self.filt_a.reshape(self.filt_a.shape[0],self.filt_a.shape[1],1)
        self.N = N
        self.passband = passband
        self.stopband = stopband
        self.gpass = gpass
        self.gstop = gstop
        self.ftype= ftype

        ParallelLinearFilterbank.__init__(self, self.filt_b, self.filt_a, samplerate)

class ButterworthFilterbank(ParallelLinearFilterbank):
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
    
    def __init__(self,samplerate, N, ord, Fn, btype='low'):
       # print Wn
        Wn=Fn.copy()
        try:
            len(Wn)
        except TypeError:
            Wn=array([Wn])
            
                
        if len(Wn)==1:
            try:
                try:
                    Wn1, Wn2 = Wn
                    Wn = (Wn1/samplerate+0.0, Wn2/samplerate+0.0)
                except TypeError,ValueError:
                    Wn = Wn/samplerate+0.0
            except DimensionMismatchError:
                raise DimensionMismatchError('Wn must be in Hz')
            self.filt_b, self.filt_a = signal.butter(ord, Wn, btype=btype)
            
            self.fs=samplerate
            self.filt_b=kron(ones((N,1)),self.filt_b)
            self.filt_b=self.filt_b.reshape(self.filt_b.shape[0],self.filt_b.shape[1],1)
            self.filt_a=kron(ones((N,1)),self.filt_a)
            self.filt_a=self.filt_a.reshape(self.filt_a.shape[0],self.filt_a.shape[1],1)
            
            
        else:
# TODO: sort that out            
#            try:
#                try:
#                    Wn1, Wn2 = Wn[i]
#                    Wn = (Wn1/samplerate+0.0, Wn2/samplerate+0.0)
#                except TypeError:
#                 Wn[i] = Wn[i]/samplerate+0.0   
#            except DimensionMismatchError:
#                raise DimensionMismatchError('Wn must be in Hz')

            self.filt_b=zeros((N,ord+1))
            self.filt_a=zeros((N,ord+1))
            for i in range((N)):
                Wn[i] = Wn[i]/samplerate*Hz+0.0
               # print Wn[i]
                #print signal.butter(ord, Wn[i], btype=btype)
                self.filt_b[i,:], self.filt_a[i,:] = signal.butter(ord, Wn[i], btype=btype)
                
        self.filt_a=self.filt_a.reshape(self.filt_a.shape[0],self.filt_a.shape[1],1)
        self.filt_b=self.filt_b.reshape(self.filt_b.shape[0],self.filt_b.shape[1],1)    
        self.N = N    
        ParallelLinearFilterbank.__init__(self, self.filt_b, self.filt_a, samplerate) 
        
           
class TimeVaryingIIRFilterbank(Filterbank):
    ''' IIR fliterbank where the coefficients vary. It is a bandpass filter
    of which the center frequency vary follwoing a OrnsteinUhlenbeck process
    '''

    @check_units(samplerate=Hz)
    def __init__(self, samplerate, coeff, m_i, s_i, tau_i):
        self.fs=samplerate
        self.N=len(m_i)
        self.b=zeros((self.N, 3, 1))
        self.a=zeros((self.N, 3, 1))
        self.t=0*ms
        self.coeff=coeff
        self.deltaT=1./self.fs
        self.m_i=m_i
        self.s_i=s_i
        self.tau_i=tau_i
        self.Q=1./coeff

        self.BW=2*arcsinh(1./2/self.Q)*1.44269 ## bandwidth in octave
        #print self.Q,self.BW
        w0=2*pi*m_i/self.fs
        self.fc=m_i
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))

        self.b[:, 0, 0]=sin(w0)/2
        self.b[:, 1, 0]=0
        self.b[:, 2, 0]=-sin(w0)/2

        #self.a=array([1 + alpha,-2*cos(w0),1 - alpha])
        self.a[:, 0, 0]=1+alpha
        self.a[:, 1, 0]=-2*cos(w0)
        self.a[:, 2, 0]=1-alpha

        #self.t=0
       ## print self.a.shape
#        self.a = a
#        self.b = b

        self.zi=zeros((self.b.shape[0], self.b.shape[1]-1, self.b.shape[2]))
        
    def timestep(self, input):
        if isinstance(input, ndarray):
            input=input.flatten()

        #self.t=self.t+self.deltaT
        #f0=8000*Hz+2000*Hz*sin(2*pi*10*Hz*self.t)
        #tau_i=100*ms
        mu_i=self.m_i/self.tau_i
        sigma_i=sqrt(2)*self.s_i/sqrt(self.tau_i)
        self.fc=self.fc-self.fc/self.tau_i*self.deltaT+mu_i*self.deltaT+sigma_i*random.randn(1)*sqrt(self.deltaT)
        BWhz=self.fc/self.Q
        #print self.fc,BWhz
        if self.fc<=50*Hz:
            self.fc=50*Hz

        if self.fc+BWhz/2>=self.fs/2:
            self.fc=self.fs/2-1000*Hz

#        if self.fc-BWhz/2<=0:
#            self.fc=BWhz/2+20*Hz
#        if self.fc+BWhz/2>=self.fs/2:
#            self.fc=self.fs/2-BWhz/2-20*Hz
#        self.fcvstime[self.t]=self.fc
#        self.t=self.t+1
        #print self.fc


        w0=2*pi*self.fc/self.fs
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))

        self.b[:, 0, 0]=sin(w0)/2
        self.b[:, 1, 0]=0
        self.b[:, 2, 0]=-sin(w0)/2

        #self.a=array([1 + alpha,-2*cos(w0),1 - alpha])
        self.a[:, 0, 0]=1+alpha
        self.a[:, 1, 0]=-2*cos(w0)
        self.a[:, 2, 0]=1-alpha
        #y=parallel_lfilter_step(self.b, self.a, input, self.zi)
        #y, self.zi = signal.lfilter(self.b, self.a, input, zi=self.zi)
        return parallel_lfilter_step(self.b, self.a, input, self.zi)

    def apply_single(self, input):
        pass
    
    def __len__(self):
        return self.N
    samplerate=property(fget=lambda self:self.fs)



if __name__=='__main__':
    from sounds import *
    from erb import *

    sound=whitenoise(100*ms).ramp()
    defaultclock.dt=1/sound.rate
    print sound.rate, erbspace(100*Hz, 5*kHz, 4)
    fb=GammatoneFilterbank(sound.rate, erbspace(100*Hz, 5*kHz, 4))
    G=FilterbankGroup(fb, sound)
    neurons=NeuronGroup(len(G), model='''dv/dt=(I-v)/(5*ms):1
                                        I:1''', reset=0, threshold=1)
    neurons.I=linked_var(G, 'output', func=lambda x:50*clip(x, 0, Inf))
    fb_mon=StateMonitor(G, 'output', record=True)
    neuron_mon=StateMonitor(neurons, 'v', record=True)
    spikes=SpikeMonitor(neurons)
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
