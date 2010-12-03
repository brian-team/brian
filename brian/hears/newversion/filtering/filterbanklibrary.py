# TODO: update all of this with the new interface/buffering mechanism
# GammatoneFilterbank already done.

from brian import *
from scipy import signal, weave, random
from filterbank import Filterbank,RestructureFilterbank
from linearfilterbank import *
from firfilterbank import *
__all__ = ['CascadeFilterbank',
           'GammatoneFilterbank',
           'ApproximateGammatoneFilterbank',
           'LogGammachirpFilterbank',
           'LinGammachirpFilterbank',
           'IIRFilterbank',
           'ButterworthFilterbank',
           'TimeVaryingIIRFilterbank',
           'Asymmetric_Compensation_Filterbank',
           'LowPassFilterbank',
           'asymmetric_compensation_coefs',
           ]

def factorial(n):
    return prod(arange(1, n+1))


class CascadeFilterbank(LinearFilterbank):
    '''
    Cascade of a filterbank (n times). 
    
    This function concatenates n times the coefficient filt_b (numerator) and filt_a (denominator)
    of a given filterbank and return another filterbank.
    
    Initialised with arguments:
    
    ``source``
        Source of the new filterbank.
        
    ``filterbank``
        Filterbank object to be put in cascade
        
    ``n``
        Number of cascades
    '''
    
    def __init__(self,source, filterbank,n):
        b=filterbank.filt_b
        a=filterbank.filt_a
        self.samplerate =  source.samplerate
        self.nchannels=filterbank.nchannels
        self.filt_b=zeros((b.shape[0], b.shape[1],n))
        self.filt_a=zeros((a.shape[0], a.shape[1],n))
        for i in range((nbr_cascade)):
            self.filt_b[:,:,i]=b[:,:,0]
            self.filt_a[:,:,i]=a[:,:,0]
            
        LinearFilterbank.__init__(self, source,self.filt_b, self.filt_a)
    
    
class GammatoneFilterbank(LinearFilterbank):
    '''
    Bank  of gammatone filters.
    
    They are implemented as cascades of four 2nd-order filters (this 8th-order digital filter corresponds to a 4th-order gammatone filter).
    
    It comes from Slaney's exact gammatone  implementation (Slaney, M., 1993, "An Efficient Implementation of the Patterson-Holdsworth 
    Auditory Filter Bank". Apple Computer Technical Report #35). The code is based on  Slaney's matlab implementation 
    (http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/)
    
    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``cf``
        List or array of center frequencies.
        
    ``b=1.019``
        parameter which determines the  bandwidth of the filters (and reciprocally the duration of its impulse response). 
        In particular, the bandwdith = b.ERB(cf), where ERB(cf) is the equivalent bandwidth at frequency ``cf``.
        The default value of ``b`` comes from Patterson et al., 1992.
        ``b`` can either be a scalar and will be the same for every channel or either an array with the same length as ``cf``
    
        
    ``erb_order=1``, ``ear_Q=9.26449`` and ``min_bw=24.7`` are parameters used to compute the ERB bandwidth. (ERB = ((cf/ear_Q)^erb_order + min_bw^erb_order)^(1/erb_order)).
    Their default values are the one recommended in Glasberg and Moore, 1990 

    '''

    def __init__(self, source, cf,b=1.019,erb_order=1,ear_Q=9.26449,min_bw=24.7):
        cf = atleast_1d(cf)
        self.cf = cf
        self.samplerate =  source.samplerate
        T = 1/self.samplerate
        self.b,self.erb_order,self.EarQ,self.min_bw=b,erb_order,ear_Q,min_bw
        erb = ((cf/ear_Q)**erb_order + min_bw**erb_order)**(1/erb_order)
        B = b*2*pi*erb
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

        self.filt_a=dstack((array([ones(len(cf)), B1, B2]).T,)*4)
        self.filt_b=dstack((array([A0/gain, A11/gain, A2/gain]).T,
                         array([A0*ones(len(cf)), A12, zeros(len(cf))]).T,
                         array([A0*ones(len(cf)), A13, zeros(len(cf))]).T,
                         array([A0*ones(len(cf)), A14, zeros(len(cf))]).T))
    
        LinearFilterbank.__init__(self, source, self.filt_b, self.filt_a)

class ApproximateGammatoneFilterbank(LinearFilterbank):
    '''
    Bank of approximate gammatone filters implemented as a cascade of n  gammatone filters..
    
    The design is based on the Hohmann implementation (hohmann, V., 2002, Frequency analysis and synthesis using a Gammatone filterbank,
    Acta Acustica United with Acustica). The code is based on the matlab gammatone implementation from the Meddis'toolbox (http://www.essex.ac.uk/psychology/psy/PEOPLE/meddis/webFolder08/WebIntro.htm) 
    
    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``cf``
        List or array of center frequencies.
        
    ``bandwidth``
        List or array of filters bandwidth corresponding, one for each cf.
        
    ``order=4``
        order is the number of 1st-order gammatone filters put in cascade and is therefore the order the resulting gammatone filters.
    
     '''
   
    def __init__(self, source, cf,  bandwidth,order=4):
        cf = atleast_1d(cf)
        bandwidth = atleast_1d(bandwidth)
        self.cf = cf
        self.samplerate =  source.samplerate
        dt = 1/self.samplerate 
        phi = 2 * pi * bandwidth * dt
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
        self.filt_a = dstack((array([b0, b1, b2]).T,)*order)
        self.filt_b = dstack((array([a0, a1, zeros(len(cf))]).T,)*order)
        self.order = order
        
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a)
        
        
class Asymmetric_Compensation_Filterbank(LinearFilterbank):
    '''
    Bank of asymmetric compensation fitlers
    
    Those filters are meant to be used in cascade with gammatone filters to approximate gammachirp filters (Unoki et al., 2001, Improvement of an IIR asymmetric compensation 
    gammachirp filter, Acoust. Sci. & Tech.). They are implemented a a cascade of low order filters. The code is based on the implementation found in the AIM-MAT toolobox 
    (http://www.pdn.cam.ac.uk/groups/cnbh/aimmanual/index.html)

    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``f``
        List or array of the cut of frequencies
        
    ``b=1.019``
        parameters which determines the duration of the impulse response
        ``b`` can either be a scalar and will be the same for every channel or either an array with the same length as ``cf``
    ``c=1``
        c is the glide slope when this filter is used to implement a gammachirp
        ``c`` can either be a scalar and will be the same for every channel or either an array with the same length as ``cf``
     ``ncascades=4``
        ncascades is the number of time the basic fitler is put in cascade.
    
     '''
     
    def __init__(self, source, f,b=1.019, c=1,ncascades=4):
        
        f = atleast_1d(f)
        self.f = f
        self.samplerate =  source.samplerate     
        ERBw=24.7*(4.37e-3*f+1.)
        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.filt_b=zeros((len(f), 3, nbr_cascade))
        self.filt_a=zeros((len(f), 3, nbr_cascade))

        for k in arange(nbr_cascade):

            r=exp(-p1*(p0/p4)**(k)*2*pi*b*ERBw/self.samplerate) #k instead of k-1 because range 0 N-1
            Df=(p0*p4)**(k)*p2*c*b*ERBw

            phi=2*pi*maximum((f+Df), 0)/self.samplerate
            psy=2*pi*maximum((f-Df), 0)/self.samplerate

            ap=vstack((ones(r.shape),-2*r*cos(phi), r**2)).T
            bz=vstack((ones(r.shape),-2*r*cos(psy), r**2)).T

            fn=f#+ compensation_filter_order* p3 *c *b *ERBw/4;

            vwr=exp(1j*2*pi*fn/self.samplerate)
            vwrs=vstack((ones(vwr.shape), vwr, vwr**2)).T

            ##normilization stuff
            nrm=abs(sum(vwrs*ap, 1)/sum(vwrs*bz, 1))
            
            bz=bz*tile(nrm,[3,1]).T
            self.filt_b[:, :, k]=bz
            self.filt_a[:, :, k]=ap

        LinearFilterbank.__init__(self, source, self.filt_b, self.filt_a)


class LowPassFilterbank(LinearFilterbank):
    '''
    Bank of 1st-order lowpass filters
    
    The code is based on the code found in the Meddis'toolbox (http://www.essex.ac.uk/psychology/psy/PEOPLE/meddis/webFolder08/WebIntro.htm). 
    It was implemented here to be used in the DRNL cochlear model implementation.

    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``fc``
        List or array (with length = number of channels) of cutoff frequencies
    '''
    def __init__(self,source,fc):
        nchannels=len(fc)
        self.samplerate= source.samplerate
        dt=1./self.samplerate

        self.filt_b=zeros((nchannels, 2, 1))
        self.filt_a=zeros((snchannels, 2, 1))
        tau=1/(2*pi*fc)
        self.filt_b[:,0,0]=dt/tau
        self.filt_b[:,1,0]=0*ones(nchannels)
        self.filt_a[:,0,0]=1*ones(nchannels)
        self.filt_a[:,1,0]=-(1-dt/tau)
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a) 



            
class LogGammachirpFilterbank(LinearFilterbank):
    '''
    Bank of gammachirp filters with a logarithmic frequency sweep
    
    The implementation  is a cascade of 4 2nd-order IIR gammatone filter 
    followed by a cascade of  ncascade 2nd-order asymmetric compensation filters as introduced in " Unoki et al. 2001, Improvement of an IIR asymmetric 
    compensation gammachirp filter". 
    

    Initialisation parameters:
    
    ``source``
        Source sound or filterbank.
        
    ``f``
        List or array of the sweep ending frequencies. (f_inst=f+c/t)
        
        
    ``b=1.019``
        parameters which determines the duration of the impulse response
        ``b`` can either be a scalar and will be the same for every channel or either an array with the same length as ``f``
        
    ``c=1``
        c is the glide slope (or sweep rate) given in Hertz/second. The trajectory of the instantaneous frequency towards f is an upchirp when c<0 and a 
        downchirp when c>0
        ``c`` can either be a scalar and will be the same for every channel or either an array with the same length as ``f``
        
    ``ncascade=4``
        number of times the asymmetric compensation filter is put in cascade; The default value comes from Unoki et al. 2001. This parameters is only used in the IIR implementation.
        
    '''
      
    def __init__(self, source, f,b=1.019,c=1,ncascade=4):
        f = atleast_1d(f)
        self.f = f
        self.samplerate= source.samplerate
        
        self.c=c
        self.b=b
        gammatone=GammatoneFilterbank(source, f,b)

        self.gammatone_filt_b=gammatone.filt_b
        self.gammatone_filt_a=gammatone.filt_a

        ERBw=24.7*(4.37e-3*f+1.)

        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.asymmetric_filt_b=zeros((len(f),3, ncascade))
        self.asymmetric_filt_a=zeros((len(f),3, ncascade))

        self.asymmetric_filt_b,self.asymmetric_filt_a=asymmetric_compensation_coefs(self.samplerate,f,self.asymmetric_filt_b,self.asymmetric_filt_a,b,c,p0,p1,p2,p3,p4)

        #concatenate the gammatone filter coefficients so that everything is in cascade in each frequency channel
        self.filt_b=concatenate([self.gammatone_filt_b, self.asymmetric_filt_b],axis=2)
        self.filt_a=concatenate([self.gammatone_filt_a, self.asymmetric_filt_a],axis=2)
        
        LinearFilterbank.__init__(self, source, self.filt_b,self.filt_a)


class LinGammachirpFilterbank(FIRFilterbank):
    '''
    Bank of gammachirp filters with linear frequency sweeps
    
    Those filters are implemented as FIR filters using  truncated time representations of  gammachirp functions as the impulse response. The impulse responses,
    which need to have the same length for every channel, have a duration of 15 times the biggest time constant. The length of the impulse response is therefore 
    15.max(time_constant).sampling_rate. The envelope can either be a gamma function or a gaussian function (Gabor filter). In the case of the gabor filter, the duration
    of the impulse response is only 12 times the biggest time constant. The impulse responses are normalized with respect to the transmitted power, i.e. their rms =1
    
    Initialisation parameters:
    
    ``source``
        Source sound or filterbank.
        
    ``f``
        List or array of the sweep starting frequencies (f_inst=f+c*t)

        
    ``time_constant``
        determines the duration of the envelope and consequently the length of the impluse response
        
    ``c=1``
        c is the glide slope (or sweep rate) given ins Hertz/second. The time-dependent instantaneous frequency is f+c*t and is therefore going upward when c>0 and downward when c<0
        ``c`` can either be a scalar and will be the same for every channel or either an array with the same length as ``f``
        
    ``phase=0``
        phase shift of the carrier
    
    ``env='gamma'``
        the envelope can either be a gamma function (env='gamma') or a gaussian function (env='gabor')
        
    Has attributes:
    
    ``length_impulse_response`` 
        number of sample if the impulse responses
        
    ``impulse_response``
        array of shape ``nchannels``X``length_impulse_response`` with each row being an impulse response for the  corresponding channel
    
    '''
    def __init__(self,source,  f,time_constant,c,phase=0,env='gamma'):
        
        self.f=f=atleast_1d(f)
        self.c=c=atleast_1d(c)
        self.phase=phase=atleast_1d(phase)
        self.time_constant=time_constant=atleast_1d(time_constant) 
        if len(time_constant)==1:
            time_constant=time_constant*ones(len(f))
        if len(c)==1:
            c=c*ones(len(f))
        if len(phase)==1:
            phase=phase*ones(len(f))
        self.samplerate= source.samplerate
        
        
        
        Tcst_max=max(time_constant)
        if env=='gamma':
            t_start=-Tcst_max*3*second
            t=arange(t_start,-4*t_start,1./self.samplerate)

        if env=='gabor':
            t_start=-Tcst*6*second
            t=arange(t_start,-t_start,1./self.samplerate)
            
        self.impulse_response=zeros((len(f),len(t)))
                                    
        for ich in xrange(len(f)):
            if env=='gamma':
                env=(t-t_start)**3*exp(-(t-t_start)/time_constant[ich])
            if env=='gabor':  
                env=exp(-(t/(2*time_constant[ich]))**2)
                
            self.impulse_response[ich,:]=env*cos(2*pi*(f[ich]*t+c[ich]/2*t**2)+phase[ich])
            self.impulse_response[ich,:]=self.impulse_response[ich,:]/sqrt(sum(self.impulse_response[ich,:]**2))    


        FIRFilterbank.__init__(self,source, self.impulse_response)


class IIRFilterbank(LinearFilterbank):
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
    
    def __init__(self, source, N, passband, stopband, gpass, gstop, ftype):
        # passband can take form x or (a,b) in Hz and we need to convert to scipy's format
        try:
            try:
                a, b=passband
                a=a/samplerate*2+0.0    # wn=1 corresponding to half the sample rate 
                b=b/samplerate*2+0.0     
                passband=[a, b]
                a+1
                b+1
            except TypeError:
                passband=passband/samplerate
                passband+1
            try:
                a, b=stopband
                a=a/samplerate*2+0.0 
                b=b/samplerate*2+0.0    
                stopband=[a, b]
                a+1
                b+1
            except TypeError:
                stopband=stopband/samplerate
                stopband+1
        except DimensionMismatchError:
            raise DimensionMismatchError('IIRFilterbank passband, stopband parameters must be in Hz')

        # now design filterbank

        self.samplerate=source.samplerate
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

        LinearFilterbank.__init__(self, self.filt_b, self.filt_a, samplerate)


class ButterworthFilterbank(LinearFilterbank):
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
    ``Fn``
        Cutoff parameter(s) in Hz, either a single value or pair for band filters.
    ``btype``
        One of 'low', 'high', 'bandpass' or 'bandstop'.
    '''

    def __init__(self,source, N, ord, Fn, btype='low'):
        # print Wn
        Wn=Fn.copy()
        Wn=atleast_1d(Wn) #Scalar inputs are converted to 1-dimensional arrays
        self.samplerate = source.samplerate
        try:
            Wn= Wn/self.samplerate *2+0.0    # wn=1 corresponding to half the sample rate   
        except DimensionMismatchError:
            raise DimensionMismatchError('Wn must be in Hz')
        
        self.filt_b=zeros((N,ord+1))
        self.filt_a=zeros((N,ord+1))
        
        if btype=='low' or btype=='high':
            if len(Wn)==1:     #if there is only one Wn value for all channel just repeat it
                self.filt_b, self.filt_a = signal.butter(ord, Wn, btype=btype)
                self.filt_b=kron(ones((N,1)),self.filt_b)
                self.filt_a=kron(ones((N,1)),self.filt_a)
            else:               #else make N different filters
                for i in xrange((N)):
                    self.filt_b[i,:], self.filt_a[i,:] = signal.butter(ord, Wn[i], btype=btype)
        else:
            if Wn.ndim==1:     #if there is only one Wn pair of values for all channel just repeat it
                self.filt_b, self.filt_a = signal.butter(ord, Wn, btype=btype)
                self.filt_b=kron(ones((N,1)),self.filt_b)
                self.filt_a=kron(ones((N,1)),self.filt_a)
            else:   
                for i in xrange((N)):
                    self.filt_b[i,:], self.filt_a[i,:] = signal.butter(ord, Wn[i,:], btype=btype)   
                
        self.filt_a=self.filt_a.reshape(self.filt_a.shape[0],self.filt_a.shape[1],1)
        self.filt_b=self.filt_b.reshape(self.filt_b.shape[0],self.filt_b.shape[1],1)    
        self.N = N    
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a) 
        
 
class TimeVaryingIIRFilterbank(Filterbank):
    ''' IIR filterbank where the coefficients vary. 
    '''
    def __init__(self, source,interval_change,vary_filter_class):
        self.samplerate  = source.samplerate
        self.vary_filter_class=vary_filter_class
                
        self.sub_buffer_length=interval_change
        self.buffer_start=-self.sub_buffer_length
        self.vary_filter_class.sub_buffer_length=self.sub_buffer_length
        self.vary_filter_class.buffer_start=self.buffer_start
        self.b,self.a=self.vary_filter_class.filt_b,self.vary_filter_class.filt_a
        
        self.N=self.b.shape[0]
        if self.N!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source,self.N)
            
        self.source=source
        Filterbank.__init__(self, source)
        self.zi=zeros((self.b.shape[0], self.b.shape[1]-1, self.b.shape[2]))
        
    def buffer_apply(self, input):
#        if isinstance(input, ndarray):
#            input=input.flatten()

#        self.vary_filter_class()
#        self.b,self.a=self.vary_filter_class.filt_b,self.vary_filter_class.filt_a
#        print input.shape
#        return apply_linear_filterbank(self.b, self.a,input, self.zi)
        buffer_length=input.shape[0]
        response=zeros((buffer_length,self.N))
        for isub_buffer in xrange(buffer_length/self.sub_buffer_length):
            self.vary_filter_class()
            self.b,self.a=self.vary_filter_class.filt_b,self.vary_filter_class.filt_a
#            print input[isub_buffer*self.sub_buffer_length:(isub_buffer+1)*self.sub_buffer_length,:].shape
            response[isub_buffer*self.sub_buffer_length:(isub_buffer+1)*self.sub_buffer_length,:]=apply_linear_filterbank(self.b, self.a,input[isub_buffer*self.sub_buffer_length:(isub_buffer+1)*self.sub_buffer_length,:], self.zi)

        return response
    
def asymmetric_compensation_coefs(samplerate,fr,filt_b,filt_a,b,c,p0,p1,p2,p3,p4):
    '''
     overhead if passing dico
     better pass filterb a or initialize the here
     better put his function inside the __call__
    give the coefficients of an asymmetric compensation filter
    It is optimized to be used as coefficients of a time varying filter
    as all the parameters that does not change are given as arguments
    '''
    ERBw=24.7*(4.37e-3*fr+1.)
    nbr_cascade=4
    for k in arange(nbr_cascade):

        r=exp(-p1*(p0/p4)**(k)*2*pi*b*ERBw/samplerate) #k instead of k-1 because range 0 N-1

        Dfr=(p0*p4)**(k)*p2*c*b*ERBw

        phi=2*pi*maximum((fr+Dfr), 0)/samplerate
        psy=2*pi*maximum((fr-Dfr), 0)/samplerate

        ap=vstack((ones(r.shape),-2*r*cos(phi), r**2)).T
        bz=vstack((ones(r.shape),-2*r*cos(psy), r**2)).T

        vwr=exp(1j*2*pi*fr/samplerate)
        vwrs=vstack((ones(vwr.shape), vwr, vwr**2)).T

        ##normilization stuff
        nrm=abs(sum(vwrs*ap, 1)/sum(vwrs*bz, 1))
        bz=bz*tile(nrm,[3,1]).T
        filt_b[:, :, k]=bz
        filt_a[:, :, k]=ap

    return filt_b,filt_a

