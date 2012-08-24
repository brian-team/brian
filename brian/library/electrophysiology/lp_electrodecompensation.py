"""
Lp offline electrode compensation method.

    Rossant et al., "A calibration-free electrode compensation method"
    J. Neurophysiol 2012
"""
from brian import *
from scipy.optimize import fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg
from scipy.signal import lfilter
from scipy import linalg
# ndtr is the cumulative distribution function of a standard Gaussian law
from scipy.special import ndtr
import time

__all__ = ["ElectrodeCompensation", "Lp_compensate",
           "find_peak_threshold", "find_spikes", "confusion_matrix",
           "get_scores", "get_trace_quality"]

'''
Equivalent linear filter
------------------------
'''
def compute_filter(A, row=0):
    """
    Compute the equivalent digital filter of a system Y(n+1)=AY(n)+X(n).
    
    From a linear discrete-time multidimensional system Y(n+1)=AY(n)+X(n),
    with X(n) a vector with X(n)[1:]==0, this function compute the vectors
    b and a of an equivalent linear 1D filter for simulating the variable
    indexed by "row". The two vectors b, a can be used in the Scipy function
    lfilter.
    """
    d = len(A)

    # compute a: characteristic polynomial of the matrix
    a = poly(A)  # with a[0]=1

    # compute b recursively
    b = zeros(d+1)
    T = eye(d)
    b[0] = T[row, 0]
    for i in range(1, d+1):
        T = a[i]*eye(d) + dot(A, T)
        b[i] = T[row, 0]
    return b, a

def simulate(eqs, I, dt, row=0):
    """
    Simulate a linear neuron model in response to an injected current using
    an equivalent linear filter.
    
    Simulate a system of neuron equations in response to an injected current I
    with sampling frequency 1/dt, using a Scipy linear filter rather than a Brian
    integrator (it's more efficient since there are no spikes). 
    The variable indexed by "row" only is simulated.
    "I" must not appear in the equations, rather, it will be injected in the 
    *first* variable only. That is, if the equations correspond to the
    differential system dY/dt=MY+I, where only the first component of I is nonzero,
    then the equations must correspond to the matrix M. 
    For instance, if the equations were dv/dt=(R*I-v)/tau, then the following
    equation must be given: "dv/dt=-v/tau", and the input current will be
    "I*R/tau".
    """
    # get M and B such that the system is dY/dy=M(Y-B)
    # NOTE: it only works if the system is invertible (det(M)!=0) !
    M, B = get_linear_equations(eqs)
    # discretization of the system: Y(n+1)=AY(n), were A=exp(M*dt).
    A = linalg.expm(M * dt)
    # compute the filter
    b, a = compute_filter(A, row=row)
    # apply the filter to the injected current
    y = lfilter(b, a, I*dt) + B[row]
    return y
    
'''
Lp Electrode Compensation method
--------------------------------
'''
class ElectrodeCompensation (object):
    # 1RC
    # the term "+ (Re/taue) * I" in the first equation is removed
    # the coefficient behind "I" is in the variable "I_coeff"
    eqs = """
            dV/dt=Re*(-Iinj)/taue : volt
            dV0/dt=(R*Iinj-V0+Vr)/tau : volt
            Iinj=(V-V0)/Re : amp
            """
    # the parameters tuple is: R, tau, Vr, Re, taue
    # this function returns Re/taue, the coefficient behind "I"
    I_coeff = lambda self, p: p[3]/p[4]

    def __init__(self, I, Vraw,
                 dt, durslice=1*second,
                 p=1.0, 
                 criterion=None, 
                 *params):
        """
        Initialize an ElectrodeCompensation instance, used to store the model
        parameters during the optimization.

        * I: injected current, 1D vector.
        * Vraw: raw (uncompensated) voltage trace, 1D vector, same length as I.
        * dt: sampling period (inverse of the sampling frequency), in second.
        * durslice=1*second: duration of each time slice, where the fit is 
          performed independently
        * p=1.0: parameter of the Lp error. p should be less than 2.
          Experimenting with this parameter is recommended.
          Use p~=1 at first, especially with difficult recordings
          Use p~0.5 with good recordings (less noise) or with biophysical model
          simulations without noise.
        * criterion: a custom error function used in the optimization. If None,
          it is the Lp error. Otherwise, it should be a function of the form
          "lambda raw, model: error", where raw and model are the raw and linear
          model membrane potential traces. For instance, the function for the 
          Lp error is: "lambda raw, model: sum(abs(raw-model)**self.p)".
          It can also be a function of the form:
          "lambda raw, model, electrode: error" in the case when one needs
          the electrode response to compute the error.
        * *params: a list of initial parameters for the optimization, in the 
          following order: R, tau, Vr, Re, taue.
        """
        self.I = I
        self.Vraw = Vraw
        self.p = p
        self.dt = dt
        self.dt_ = float(dt)
        self.x0 = self.params_to_vector(*params)
        self.duration = len(I) * dt
        self.durslice = min(durslice, self.duration)
        self.slicesteps = int(durslice/dt)
        self.nslices = int(ceil(len(I)*dt/durslice))
        
        if criterion is None:
            self.criterion = lambda raw, model: self.Lp_error(raw, model)
        else:
            self.criterion = criterion
        
        self.islice = 0
        self.I_list = [I[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]
        self.Vraw_list = [Vraw[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]

    def vector_to_params(self, *x):
        """
        Convert a vector of parameters (used for the optimization) to a tuple
        of actual parameters.
        """
        R,tau,Vr,Re,taue = x

        # parameter transformation to force all parameters to be positive
        # and tau>taue
        R = R*R
        Re = Re*Re
        taue = taue*taue
        tau = taue + tau*tau  # taudiff
        return R*ohm,tau*second,Vr*volt,Re*ohm,taue*second

    def params_to_vector(self, *params):
        """
        Inverse function of vector_to_params: from a tuple of actual parameters,
        return a vector of parameters in the optimization space coordinates.
        """
        x = [sqrt(params[0]),
             sqrt(params[1]-params[4]),  # taudiff
             params[2],
             sqrt(params[3]),
             sqrt(params[4]),
             ]
        return list(x)

    def get_model_trace(self, row, *x):
        """
        Compute the model response (variable index "row") to the injected 
        current, at a specific slice (stored in self.islice), with model 
        parameters specified with the vector x.
        """
        # get the actual model parameters
        params = self.vector_to_params(*x)
        R, tau, Vr, Re, taue = params
        
        # get the coefficient behind I in the equations
        coeff = self.I_coeff(params)
        
        # compile the neuron equations, i.e., inject the parameters in them
        eqs = Equations(self.eqs)
        eqs.prepare()
        self._eqs = eqs
        
        # simulate the neuron response
        y = simulate(eqs, self.I_list[self.islice] * coeff, self.dt, row=row)
        return y

    def get_trace(self, islice, *params):
        """
        Get the neuron and electrode traces, in slice number "islice", with
        the list of parameters in "params" given in the order:
        R, tau, Vr, Re, taue.
        """
        x = self.params_to_vector(*params)
        
        # full model trace (neuron and electrode)
        V = self.get_model_trace(0, *x)
        
        # neuron voltage
        V0 = self.get_model_trace(1, *x)
        
        # electrode voltage
        Velec = V-V0
        
        return V0, Velec
        
    def Lp_error(self, raw, model):
        """
        Default error function: return the L^p error between the raw trace,
        and the full model trace (neuron and electrode)
        """
        return (self.dt_*sum(abs(raw-model)**self.p))**(1./self.p)
    
    def fitness(self, x):
        """
        fitness function provided to the fmin optimization procedure.
        Simulate the model and compute the error between the full model
        response (neuron and electrode) and the raw trace.
        """
        # get the parameters
        R, tau, Vr, Re, taue = self.vector_to_params(*x)
        
        # compute the full model trace
        vmodel = self.get_model_trace(0, *x)
        
        # check if the error function requests the electrode trace
        if self.criterion.func_code.co_argcount>=3:
            v0 = self.get_model_trace(1, *x)
            velec = vmodel - v0
            # call the error function with the parameters: raw, model, electrode
            e = self.criterion(self.Vraw_list[self.islice], vmodel, velec)
        else:
            # call the error function with the parameters: raw, model
            e = self.criterion(self.Vraw_list[self.islice], vmodel)
        return e

    def compensate_slice(self, x0):
        """
        Compensate on the current slice, by calling fmin on the fitness function.
        """
        fun = lambda x: self.fitness(x)
        x = fmin(fun, x0, maxiter=10000, maxfun=10000, disp=True)
        return x

    def compensate(self):
        """
        Compute compensate_slice for all slices.
        """
        # params_list contains the best parameters for each slice
        self.params_list = []
        # xlist contains the best vector for each slice
        self.xlist = [self.x0]
        t0 = time.clock()
        for self.islice in range(self.nslices):
            newx = self.compensate_slice(self.x0)
            self.xlist.append(newx)
            self.params_list.append(self.vector_to_params(*newx))
            msg = "Slice %d/%d compensated in %.2f seconds" %  \
                (self.islice+1, self.nslices, time.clock()-t0)
            log_info("electrode_compensation", msg)
            t0 = time.clock()
        self.xlist = self.xlist[1:]
        return self.xlist

    def get_compensated_trace(self):
        """
        Once the optimization is done, compute all the model traces (full, 
        neuron, electrode voltages). This function returns the compensated trace
        (raw - neuron model voltage).
        """
        Vcomp_list = []
        Vneuron_list = []
        Velec_list = []
        
        for self.islice in range(self.nslices):
            x = self.xlist[self.islice]
            V = self.get_model_trace(0, *x)
            V0 = self.get_model_trace(1, *x)
            Velec = V-V0
            
            Vneuron_list.append(V0)
            Velec_list.append(Velec)
            Vcomp_list.append(self.Vraw_list[self.islice] - Velec)
            
        self.Vcomp = hstack(Vcomp_list)
        self.Vneuron = hstack(Vneuron_list)
        self.Velec = hstack(Velec_list)
        
        return self.Vcomp

def Lp_compensate(I, Vraw, dt, 
               slice_duration=1*second,
               p=1.0,
               criterion=None,
               full=False, docompensation=True,
               **initial_params):
    """
    Perform the L^p electrode compensation technique on a recorded membrane
    potential.
    
    * I: injected current, 1D vector.
    * Vraw: raw (uncompensated) voltage trace, 1D vector, same length as I.
    * dt: sampling period (inverse of the sampling frequency), in second.
    * slice_duration=1*second: duration of each time slice, where the fit is 
      performed independently
    * p=1.0: parameter of the Lp error. p should be less than 2.
      Experimenting with this parameter is recommended.
      Use p~1 at first, especially with difficult recordings
      Use p~0.5 with good recordings (less noise) or with biophysical model
      simulations without noise.
    * criterion: a custom error function used in the optimization. If None,
      it is the Lp error. Otherwise, it should be a function of the form
      "lambda raw, model: error", where raw and model are the raw and linear
      model membrane potential traces. For instance, the function for the 
      Lp error is: "lambda raw, model: sum(abs(raw-model)**self.p)".
      It can also be a function of the form:
      "lambda raw, model, electrode: error" in the case when one needs
      the electrode response to compute the error.
    * full=False: if False, return a tuple (compensated_trace, parameters)
      where parameters is an array of the best parameters (one column/slice)
      If True, return a dict with the following keys:
      Vcompensated, Vneuron, Velectrode, params=params, instance
      where instance in the ElectrodeCompensation object.
    * docompensation=True: if False, does not perform the optimization and only
      return an ElectrodeCompensation object instance, to take full control over
      the optimization procedure.
    * params: a list of initial parameters for the optimization, in the 
      following order: R, tau, Vr, Re, taue. Best results are obtained when
      reasonable estimates of the parameters are given.
    """
    R = initial_params.get("R", 100*Mohm)
    tau = initial_params.get("tau", 20*ms)
    Vr = initial_params.get("Vr", -70*mV)
    Re = initial_params.get("Re", 50*Mohm)
    taue = initial_params.get("taue", .5*ms)
    
    comp = ElectrodeCompensation(I, Vraw,
                                 dt,
                                 slice_duration,
                                 p, criterion,
                                 R, tau, Vr, Re, taue,
                                 )
    if docompensation:
        comp.compensate()
        Vcomp = comp.get_compensated_trace()
        params = array(comp.params_list).transpose()
        if not full:
            return Vcomp, params
        else:
            return dict(Vcompensated=Vcomp, Vneuron=comp.Vneuron,
                        Vfull=comp.Vneuron+comp.Velec,
                        Velectrode=comp.Velec, params=params, instance=comp)
    else:
        return dict(instance=comp)
        
'''
Automatic spike detection
-------------------------
'''
def find_peaks(v):
    """
    Return the indices of the local extrema on a trace v.
    """
    dv = diff(v)
    peaks = ((dv[1:] * dv[:-1]) <= 0).nonzero()[0] + 1
    return peaks
    
def find_peak_threshold(v, nbins=20, full=False):
    """
    Find an adequate separatrix for spike detection in an intracellular trace.
    
    This threshold is found through the following procedure:
    * find all peaks in the trace, corresponding to a crossing of dv=0 
      in the phase space. Those peaks approximately follow a mixture of two
      gaussian distributions, one containg all spikes, the other one containing 
      non relevant peaks in the subthreshold trace.
    * compute an histogram of the peak values and find the boundary between
      the two gaussian distributions, by finding a relevant local minimum in
      this histogram.
    
    Possible improvements:
    * use a kernel density estimation
    * try the mean-shift algorithm for the clustering step
    
    Parameters:
    * nbins=20: number of bins in the histogram
    * full=False: if True, return a dict with the following keys:
      separatrix, peak_values, peaks, allminima, histogram, bins.
      else, return
    """
    peaks = find_peaks(v)
    vc = v[peaks] # peak values
    vc0 = vc.copy()
    
    m = median(vc)
    
    # histogram of the peaks
    bins = linspace(m, max(vc), nbins)
    h0,b = histogram(vc, bins)
    h = h0[:-5]
    dh = diff(h)

    # all local minima
    candidates = nonzero((dh[:-1]<=0) & (dh[1:]>=0))[0]+1

    try:
        if len(candidates)>1:
            # find the longest sequence of successive integers in candidates
            dcandidates = diff(candidates)
            dcandidates[dcandidates!=1] = 0
            # the last index of the longest sequence
            i1 = argmax(cumsum(dcandidates))+1
            # first index of this sequence
            i0 = nonzero(dcandidates[:i1+1]!=1)[0]
            if len(i0)>0: i0 = i0[0]
            else: i0 = 0
            vc = mean(b[candidates[i0:i1+1]])
        else:
            vc = b[candidates[0]]
    except:
        vc = (m+max(vc))/2
    
    if full:
        # return all variables
        return dict(separatrix=vc, peak_values=vc0, allminima=candidates, 
                    histogram=h0, bins=b,peaks=peaks)
    # return just the separatrix
    return vc
    
def find_spikes(v, vc=None, dt=0.1*ms, refractory=5*ms, check_quality=False):
    """
    Find spikes in an intracellular trace.
    
    * vc=None: separatrix (in volt). If None, a separatrix will be automatically 
      detected using the method described in the paper.
    * dt=0.1*ms: timestep in the trace (inverse of the sampling frequency)
    * refractory=5*ms: refractory period: minimal duration between two 
      successive spikes
    * check_quality=False: if True, will check spike detection quality using
      signal detection theory. The function then returns a tuple (spikes,scores)
      where scores is a dict.
    """
    # compute the refractory period in number of time steps
    refractory = int(refractory/dt)
    # determine the separatrix automatically
    if vc is None:
        vc = find_peak_threshold(v)
    dv = diff(v)
    spikes = ((v[1:] > vc) & (v[:-1] < vc)).nonzero()[0]
    spikepeaks = []
    if len(spikes) > 0:
        for i in range(len(spikes) - 1):
            # the peak is the max of [spike, spike+refractory]
            # where spike is the time when v crosses vc
            spike = spikes[i] + argmax(v[spikes[i]:spikes[i]+refractory])
            # refractory period: discard spurious spikes
            if len(spikepeaks)>0:
                if (spike-spikepeaks[-1]<refractory):
                    continue
            spikepeaks.append(spike)
        decreasing = (dv[spikes[-1]:] <= 0).nonzero()[0]
        if len(decreasing) > 0:
            spikepeaks.append(spikes[-1] + decreasing[0])
        else:
            spikepeaks.append(len(dv)) # last element
    spikes = array(spikepeaks)
    
    if check_quality:
        TP, FP, FN, TN = confusion_matrix(v, vc)
        scores = get_scores(TP, FP, FN, TN)
        MCC = scores['MCC']
        print "MCC: %.8f" % MCC
        return spikes, scores
    else:
        return spikes
    
def confusion_matrix(v, vc, full=False):
    """
    Compute the confusion matrix of detected spikes vs. non-spike local extrema.
    
    * v: intracellular trace
    * vc: separatrix
    * full=False: if True, return a dict with keys:
      TP, FP, FN, TN, mu1, s1, mu2, s2
    """
    sign_changes = find_peaks(v)
    peaks = v[sign_changes] # peak values
    peaks = sort(peaks) # sort values
        
    v1 = v2 = vc
    # first cluster
    mu1 = mean(peaks[peaks<=v1])
    s1 = std(peaks[peaks<=v1])
    # second cluster
    mu2 = mean(peaks[peaks>=v2])
    s2 = std(peaks[peaks>=v2])
    # compute the confusion matrix
    TP = 1-ndtr((v2-mu2)/s2)
    FP = 1-ndtr((v2-mu1)/s1)
    FN = ndtr((v1-mu2)/s2)
    TN = ndtr((v1-mu1)/s1)
    if full:
        return dict(TP=TP,FP=FP,FN=FN,TN=TN,mu1=mu1,s1=s1,mu2=mu2,s2=s2)
    return TP, FP, FN, TN
    
def get_scores(TP, FP, FN, TN):
    """
    Compute various scores associated with the confusion matrix.
    
    Return a dict with values: 
    TP, FP, FN, TN, TPR, FPR, ACC, MCC, F1
    """
    P = TP+FN
    N = FP+TN
    P2 = TP+FP
    N2 = FN+TN
    TPR = TP/P # true positive rate
    FPR = FP/N # false positive rate
    C = array([[TP,FP],[FN,TN]]) # confusion matrix

    # ACCuracy
    ACC = (TP+TN)/(P+N)

    # MCC (coef de Matthews)
    MCC = (TP*TN-FP*FN)/sqrt(P*N*P2*N2)

    # F1 score
    F1 = 2*TP/(P+P2)
    
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                TPR=TPR, FPR=FPR,ACC=ACC,MCC=MCC,F1=F1)


'''
Quality criterion
-----------------
'''
def current_before_spikes(v, I, spikes, full=False):
    """
    Compute the voltage and current before each spikes.
    
    * v: raw voltage
    * I: injected current
    * spikes: spikes indices
    * full=False: if True, return a dict with keys:
      coefficients, after_onsets, spike_before, spike_onset, spike_after
    """
    # pre-spike phase between spike_before and spike_after, relative to spike
    # peak.
    spike_before = -100
    spike_after = 10
    # onset of spike relative to spike peak
    spike_onset = -20
    t_before = arange(spike_onset-spike_before)
    after_onsets = []
    # list of pairs (a, b) of the linear regression of voltage depolarization
    # before each spike
    cs = []
    for spike in spikes:
        bef = spike + spike_before
        onset = spike + spike_onset
        after = spike + spike_after
        # v just before the spike
        before = v[bef:onset]
        # mean i during the AP
        after_onset = mean(I[onset:after])
        after_onsets.append(after_onset)
        # linear regression I -> peak
        c = polyfit(t_before, before, 1)
        cs.append(c)
    cs = array(cs) 
    after_onsets = array(after_onsets) 
    if not(full):
        return cs, after_onsets
    else:
        return dict(coefficients=cs, after_onsets=after_onsets, spike_before=spike_before,
                    spike_onset=spike_onset, spike_after=spike_after,)
    
def best_peak_prediction(cs, peaks):
    """
    Compute the best linear prediction of the peak from the linear regression c
    """
    A = array([[sum(cs[:,0]**2), sum(cs[:,0]*cs[:,1])], \
                [sum(cs[:,0]*cs[:,1]), sum(cs[:,1]**2)]])
    B = array([[sum(cs[:,0]*peaks)], [sum(cs[:,1]*peaks)]])
    u = linalg.solve(A, B)
    peaks_prediction = dot(u.reshape((1, -1)), cs.transpose())
    return peaks_prediction.flatten()
    
def get_trace_quality(v, I, full=False):
    """
    Compute the quality of a compensated trace.
    
    * v: a compensated intracellular trace
    * I: injected current
    * full=False: if True, return a dict with the following keys:
      correlation, spikes, coefficients, after_onsets, peaks_prediction,
      after_onsets, spike_before, spike_onset, spike_after
    """
    peaks = find_peaks(v)
    vc = find_peak_threshold(v)
    spikes = find_spikes(v,vc)
    # linear regression of current before spikes
    r = current_before_spikes(v, I, spikes, True)
    cs, after_onsets = r["coefficients"], r["after_onsets"]
    # compute the best linear prediction of the peak from the linear regression
    peaks_prediction = best_peak_prediction(cs, v[spikes])
    c = abs(corrcoef(v[spikes] - peaks_prediction, after_onsets)[0,1])
    if not(full):
        return c
    else:
        return dict(dict(correlation=c, spikes=spikes, cs=cs,
        after_onsets=after_onsets,
        peaks_prediction=peaks_prediction).items()+r.items())
    