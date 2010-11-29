# TODO: update all of this with the new interface/buffering mechanism
# GammatoneFilterbank already done.

from brian import *
from scipy import signal, weave, random
from filterbank import Filterbank,RestructureFilterbank
from linearfilterbank import *

__all__ = ['CascadeFilterbank',
           'GammatoneFilterbank',
           'MeddisGammatoneFilterbank',
           'GammachirpIIRFilterbank',
           'GammachirpFIRFilterbank',
           'IIRFilterbank',
           'ButterworthFilterbank',
           'TimeVaryingIIRFilterbank',
           'Asym_Comp_Filterbank',
           'LowPassFilterbank',
           'Asym_Comp_Coeff',
           ]

def factorial(n):
    return prod(arange(1, n+1))

# TODO: in the new version, this filterbank should replace the one it is applied
# to, meaning that its source should be the source of the filterbank it is being
# applied to.
class CascadeFilterbank(LinearFilterbank):
    '''
    Cascade of a filterbank (nbr_cascade times)
    '''
    
    def __init__(self,source, filterbank,nbr_cascade):
        b=filterbank.filt_b
        a=filterbank.filt_a
        self.fs = fs = source.samplerate
        self.N=filterbank.N
        self.filt_b=zeros((b.shape[0], b.shape[1],nbr_cascade))
        self.filt_a=zeros((a.shape[0], a.shape[1],nbr_cascade))
        for i in range((nbr_cascade)):
            self.filt_b[:,:,i]=b[:,:,0]
            self.filt_a[:,:,i]=a[:,:,0]
            
        LinearFilterbank.__init__(self, source,self.filt_b, self.filt_a)
    
    
class GammatoneFilterbank(LinearFilterbank):
    '''
    Exact gammatone based on Slaney's Auditory Toolbox for Matlab
    
    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
    ``fc``
        List or array of center frequencies.
    
    The ERBs are computed based on parameters in the Auditory Toolbox.
    
    TODO: improve documentation.
    '''
    # Change the following three parameters if you wish to use a different
    # ERB scale.  Must change in ERBSpace too.
    EarQ=9.26449                #  Glasberg and Moore Parameters
    minBW=24.7
    order=1
    
    @check_units(fs=Hz)
    def __init__(self, source, cf,b=None):

        cf = array(cf)
        self.cf = cf
        self.N = len(cf)
        self.fs = fs = source.samplerate
        fs = float(fs)
        EarQ, minBW, order = self.EarQ, self.minBW, self.order
        T = 1/fs
        ERB = ((cf/EarQ)**order + minBW**order)**(1/order)
        if b==None:
            B = 1.019*2*pi*ERB
        else:
            B = b*2*pi*ERB
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

        self.filt_a=dstack((array([ones(len(cf)), B1, B2]).T,)*4)
        self.filt_b=dstack((array([A0/gain, A11/gain, A2/gain]).T,
                         array([A0*ones(len(cf)), A12, zeros(len(cf))]).T,
                         array([A0*ones(len(cf)), A13, zeros(len(cf))]).T,
                         array([A0*ones(len(cf)), A14, zeros(len(cf))]).T))
    
        LinearFilterbank.__init__(self, source, self.filt_b, self.filt_a)

class MeddisGammatoneFilterbank(LinearFilterbank):
    '''
    Parallel version of Ray Meddis' UTIL_gammatone.m
    '''
    # These consts from Hohmann gammatone code
    EarQ = 9.26449                #  Glasberg and Moore Parameters
    minBW = 24.7
    
    @check_units(fs=Hz)
    def __init__(self, source, cf, order, bw):  
        cf = array(cf)
        bw = array(bw)
        self.cf = cf
        self.N = len(cf)
        self.fs = fs = source.samplerate
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
        self.filt_a = dstack((array([b0, b1, b2]).T,)*order)
        self.filt_b = dstack((array([a0, a1, zeros(len(cf))]).T,)*order)
        self.order = order
        
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a)
        
        
class Asym_Comp_Filterbank(LinearFilterbank):

     
    def __init__(self, source, fr, c=None,asym_comp_order=None,b=None):
        fr = array(fr)
        ''' 
        TODO difference between order and cascade
        '''
        
        self.fr = fr
        self.N = len(fr)
     
        self.fs = fs = source.samplerate
        if c==None:
            c=1*ones((fr.shape))
        if b==None:
            b=1.019*ones((fr.shape))
        if asym_comp_order==None:
            order=3
        compensation_filter_order=4
        ERBw=24.7*(4.37e-3*fr+1.)
        nbr_cascade=4
        order=1
        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.filt_b=zeros((len(fr), 2*order+1, nbr_cascade))
        self.filt_a=zeros((len(fr), 2*order+1, nbr_cascade))

        for k in arange(nbr_cascade):

            r=exp(-p1*(p0/p4)**(k)*2*pi*b*ERBw/self.fs) #k instead of k-1 because range 0 N-1

            Dfr=(p0*p4)**(k)*p2*c*b*ERBw

            phi=2*pi*maximum((fr+Dfr), 0)/self.fs
            psy=2*pi*maximum((fr-Dfr), 0)/self.fs

            ap=vstack((ones(r.shape),-2*r*cos(phi), r**2)).T
            bz=vstack((ones(r.shape),-2*r*cos(psy), r**2)).T

            fn=fr#+ compensation_filter_order* p3 *c *b *ERBw/4;

            vwr=exp(1j*2*pi*fn/self.fs)
            vwrs=vstack((ones(vwr.shape), vwr, vwr**2)).T

            ##normilization stuff
            nrm=abs(sum(vwrs*ap, 1)/sum(vwrs*bz, 1))
            
            bz=bz*tile(nrm,[3,1]).T
            self.filt_b[:, :, k]=bz
            self.filt_a[:, :, k]=ap

        LinearFilterbank.__init__(self, source, self.filt_b, self.filt_a)

class LowPassFilterbank(LinearFilterbank):
   
    def __init__(self,source, N,cutOffFrequency,nbr_cascade=None):
        self.N=N
        self.fs = fs = source.samplerate
        dt=1./self.fs
        if nbr_cascade==None:
            self.filt_b=zeros((self.N, 2, 1))
            self.filt_a=zeros((self.N, 2, 1))
            nbr_cascade=1
        else:
            self.filt_b=zeros((self.N, 2, nbr_cascade))
            self.filt_a=zeros((self.N, 2, nbr_cascade))
            
        tau=1/(2*pi*cutOffFrequency)
        for icascade in xrange(nbr_cascade):
            self.filt_b[:,0,icascade ]=dt/tau*ones(N)
            self.filt_b[:,1,icascade ]=0*ones(N)
            self.filt_a[:,0,icascade ]=1*ones(N)
            self.filt_a[:,1,0]=-(1-dt/tau)*ones(N)
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a) 
        
 
            
class GammachirpIIRFilterbank(LinearFilterbank):
    '''
    Implementaion of the gammachirp filter with logarithmic chirp as a cascade of a 4 second order IIR gammatone filter 
    and a 4 second orders asymmetric compensation filters
    From Unoki et al. 2001, Improvement of an IIR asymmetric compensation gammachirp filter,  
     
     comment: no GPU implementation so far... because
     c determines the rate of the frequency modulation or the chirp rate
     center_frequency 
     fr is the center frequency of the gamma tone (note: it is note the peak frequency of the gammachirp)
     '''
     
     
    def __init__(self, source, fr, c=None,asym_comp_order=None,b=None):
        fr = array(fr)

        self.fr = fr
        self.N = len(fr)


        self.fs = fs = source.samplerate
        if c==None:
            c=1*ones((fr.shape))
        if b==None:
            b=1.019*ones((fr.shape))
        if asym_comp_order==None:
            order=3
            
        self.c=c
        self.b=b
        gammatone=GammatoneFilterbank(source, fr,b)
        order=gammatone.order

        self.gammatone_filt_b=gammatone.filt_b
        self.gammatone_filt_a=gammatone.filt_a

        ERBw=24.7*(4.37e-3*fr+1.)
        compensation_filter_order=4


        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.asymmetric_filt_b=zeros((len(fr), 2*order+1, 4))
        self.asymmetric_filt_a=zeros((len(fr), 2*order+1, 4))

        self.asymmetric_filt_b,self.asymmetric_filt_a=Asym_Comp_Coeff(self.fs,fr,self.asymmetric_filt_b,self.asymmetric_filt_a,b,c,order,p0,p1,p2,p3,p4)

        #print B.shape,A.shape,Btemp.shape,Atemp.shape    
        #concatenate the gammatone filter coefficients so that everything is in cascade in each frequency channel
        #print self.gammatone_filt_b,self.asymmetric_filt_b
        self.filt_b=concatenate([self.gammatone_filt_b, self.asymmetric_filt_b],axis=2)
        self.filt_a=concatenate([self.gammatone_filt_a, self.asymmetric_filt_a],axis=2)

        LinearFilterbank.__init__(self, source,self.filt_b, self.filt_a)

class GammachirpFIRFilterbank(LinearFilterbank):
    '''
    Fit of a auditory filter (from a reverse correlation) at the NM of a barn owl at 4.6 kHz. The tap of the FIR filter
    are the time response of the filter which is long. It is thus very slow ( at least without GPU)
    The response is normalized so that every parameter set give the same peak value
    
    '''
    def __init__(self,source, fs, F0,c,time_constant):
        try:
            len(F0)
            len(c)
            len(time_constant)
        except TypeError:
            F0=array([F0])
            c=array([c])
            time_constant=array([time_constant])
            
        F0=F0/1000
        c=c/1000000
        time_constant=time_constant*1000
        fs=float(fs)
        F0 = array(F0)
        self.F0 = F0
        self.N = len(F0)
        self.fs = fs

        #%x = [amplitude, delay, time constant, frequency, phase, bias, IF glide slope]

        x=array([0.8932, 0.7905 , 0.3436  , 4.6861  ,-4.4308 ,-0.0010  , 0.3453])
        t=arange(0, 4, 1./fs*1000)
        
        LenGC=len(t)
        filt_b=zeros((len(F0), LenGC, 1))
        filt_a=zeros((len(F0), LenGC, 1))
        g=4
        for i_channel in xrange(len(F0)):  
            
            x[-1]=c[i_channel]
            x[2]=time_constant[i_channel]
            x[3]=F0[i_channel]
            #x=array([0.8932, 0.7905 , 0.3436  , 4.6861  ,-4.4308 ,-0.0010  , 0.3453])
            tmax=x[2]*(g-1)
            G=x[0]/(tmax**(g-1)*exp(1-g))*(t-x[1]+tmax)**(g-1)*exp(-(t-x[1]+tmax)/x[2])*cos(2*pi*(x[3]*(t-x[1])+x[6]/2*(t-x[1])**2)+x[4])+x[5]
            G=G*(t-x[1]+tmax>0)
            G=G/max(G)/26
#            plot(t,G)
#            show()
#            exit()
            filt_b[i_channel, :, 0]=G
            filt_a[i_channel, 0, 0]=1

        LinearFilterbank.__init__(self, filt_b, filt_a, fs*Hz)


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
    
    def __init__(self, samplerate, N, passband, stopband, gpass, gstop, ftype):
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
    ``Wn``
        Cutoff parameter(s) in Hz, either a single value or pair for band filters.
    ``btype``
        One of 'lowpass', 'highpass', 'bandpass' or 'bandstop'.
    '''

    def __init__(self,source, N, ord, Fn, btype='low'):
       # print Wn
        Wn=Fn.copy()
        Wn=atleast_1d(Wn) #Scalar inputs are converted to 1-dimensional arrays
        self.fs = fs = source.samplerate
        try:
            Wn= Wn/self.fs*2+0.0    # wn=1 corresponding to half the sample rate   
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
    ''' IIR fliterbank where the coefficients vary. It is a bandpass filter
    of which the center frequency vary follwoing a OrnsteinUhlenbeck process
    '''

    @check_units(samplerate=Hz)
    def __init__(self, source,interval_change,vary_filter_class):
        
        
        self.fs = fs = source.samplerate
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
    
def Asym_Comp_Coeff(samplerate,fr,filt_b,filt_a,b,c,order,p0,p1,p2,p3,p4):
    '''
     overhead if passing dico
     better pass filterb a or initiliaze the here
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

