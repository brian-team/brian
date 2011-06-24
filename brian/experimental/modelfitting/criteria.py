from brian import Equations, NeuronGroup, Clock, CoincidenceCounter, Network, zeros, array, \
                    ones, kron, ms, second, concatenate, hstack, sort, nonzero, diff, TimedArray, \
                    reshape, sum, log, Monitor, NetworkOperation, defaultclock, linspace, vstack, \
                    arange, sort_spikes, rint, SpikeMonitor, Connection, int32, double,SpikeGeneratorGroup,DelayConnection,\
                    SpikeCounter,forget,diagflat,inf,sqrt,ones_like,isnan,append,repeat,tile,mean,logical_and,where
from brian.tools.statistics import firing_rate, get_gamma_factor
from playdoh import *
try:
    import pycuda
    import pycuda.driver as drv
    import pycuda
    from pycuda import gpuarray
    can_use_gpu = True
except ImportError:
    can_use_gpu = False
from brian.experimental.codegen.integration_schemes import *
import sys, cPickle




class Criterion(Monitor, NetworkOperation):
    """
    Abstract class from which modelfitting criterions should derive.
    Derived classes should implement the following methods:
    
    ``initialize(self, **params)``
        Called once before the simulation. ```params`` is a dictionary with criterion-specific
        parameters. 
    
    ``timestep_call(self)``
        Called at every timestep.
    
    ``spike_call(self, neurons)``
        Called at every spike, with the spiking neurons as argument.
    
    ``get_values(self)``
        Called at the end, returns the criterion values.
        
    You have access to the following methods:
    
    ``self.get_value(self, varname)``
        Returns the value of the specified variable for all neurons (vector).
    
    You have access to the following attributes:
    
    ``self.step``
        The time step (integer)
    
    ``self.group``
        The NeuronGroup
    
    ``self.traces=None``
        Target traces.
        A 2-dimensional K*T array where K is the number of targets, and T the total number of timesteps.
        It is still a 2-dimensional array when K=1. It is None if not specified.
        
    ``self.spikes=None``
        A list of target spike trains : [(i,t)..] where i is the target index and t the spike time.
    
    ``self.N``
        The number of neurons in the NeuronGroup
    
    ``self.K=1``
        The number of targets.
    
    ``self.duration``
        The total duration of the simulation, in seconds.
    
    ``self.total_steps``
        The total number of time steps.
    
    ``self.dt``
        The timestep duration, in seconds.
    
    ``self.delays=zeros(self.n)``
        The delays for every neuron, in seconds. The delay is relative to the target.
    
    ``self.onset=0*ms``
        The onset, in seconds. The first timesteps, before onset, should be discarded in the criterion.
    
    ``self.intdelays=0``
        The delays, but in number of timesteps (``int(delays/dt)``).
    
    NOTE: traces and spikes have all been sliced before being passed to the criterion object
    """
    def __init__(self, group, traces=None, spikes=None, targets_count=1, duration=None, onset=0*ms, 
                 spikes_inline=None, spikes_offset=None,trials_offset=None,
                 traces_inline=None, traces_offset=None,
                 delays=None, when='end', **params):
        NetworkOperation.__init__(self, None, clock=group.clock, when=when)
        self.group = group
        # needed by SpikeMonitor
        self.source = group
        self.source.set_max_delay(0)
        self.delay = 0
        self.trials_offset=trials_offset
        self.N = len(group) # number of neurons
        self.K = targets_count # number of targets
        self.dt = self.clock.dt
        if traces is not None:
            self.traces = array(traces) # KxT array
            if self.traces.ndim == 1:
                self.traces = self.traces.reshape((1,-1))
            assert targets_count==self.traces.shape[0]
        self.spikes = spikes
        # get the duration from the traces if duration is not specified in the constructor
        if duration is None: duration = self.traces.shape[1] # total number of steps
        self.duration = duration
        self.total_steps = int(duration/self.dt)
        if delays is None: delays = zeros(self.n)
        self.delays = delays
        self.onset = int(onset/self.dt)
        self.intdelays = array(self.delays/self.clock.dt, dtype=int)
        self.mindelay = min(delays)
        self.maxdelay = max(delays)
        # the following data is sliced
        self.spikes_inline = spikes_inline
        self.spikes_offset = spikes_offset
        self.traces_inline = traces_inline
        self.traces_offset = traces_offset
        if self.spikes is not None:
            # target spike count and rates
            self.target_spikes_count = self.get_spikes_count(self.spikes)
            self.target_spikes_rate = self.get_spikes_rate(self.spikes)
        self.initialize(**params)

    def step(self):
        """
        Return the current time step
        """
        return int(round(self.clock.t/self.dt))
    
    def get_spikes_count(self, spikes):
        
        count = zeros(self.K)
        for (i,t) in spikes:
            count[i] += 1
        return count
    
    def get_spikes_rate(self, spikes):
        count = self.get_spikes_count(spikes)
        return count*1.0/self.duration
    
    def get_gpu_code(self): # TO IMPLEMENT
        """
        Returns CUDA code snippets to put at various places in the CUDA code template.
        It must return a dictionary with the following items:
            %CRITERION_DECLARATION%: kernel declaration code
            %CRITERION_INIT%: initialization code
            %CRITERION_TIMESTEP%: main code, called at every time step
            %CRITERION_END%: finalization code
        """
        log_warn("GPU code not implemented by the derived criterion class")
    
    def initialize_cuda_variables(self): # TO IMPLEMENT
        """
        Initialize criterion-specific CUDA variables here.
        """
        pass
    
    def get_kernel_arguments(self): # TO IMPLEMENT
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        pass
    
    def initialize(self): # TO IMPLEMENT
        """
        Override this method to initialize the criterion before the simulation
        """
        pass
    
    def timestep_call(self): # TO IMPLEMENT
        """
        Override this method to do something at every time step
        """
        pass
    
    def spike_call(self, neurons): # TO IMPLEMENT
        """
        Override this method to do something at every time spike. neurons contains
        the list of neurons that just spiked
        """
        pass
    
    def get_value(self, varname):
        return self.group.state_(varname)
    
    def __call__(self):
        self.timestep_call()
        neurons = self.group.get_spikes()
        self.spike_call(neurons)
        
#    def propagate(self, neurons):
#        self.spike_call(neurons)

    def get_values(self): # TO IMPLEMENT
        """
        Override this method to return the criterion values at the end.
        It must return one (or several, as a tuple) additive values, i.e.,
        values corresponding to slices. The method normalize can normalize
        combined values.
        """
        pass
    additive_values = property(get_values)

    def normalize(self, values): # TO IMPLEMENT
        """
        Values contains combined criterion values. It is either a vector of values (as many values
        as neurons), or a tuple with different vector of as many values as neurons.
        """
        pass

class CriterionStruct(object):
    type = None # 'trace', 'spike' or 'both'
    def get_name(self):
        return self.__class__.__name__
    name = property(get_name)




class LpErrorCriterion(Criterion):
    type = 'traces'
    def initialize(self, p=2, varname='v', method='all',insets=None,outsets=None,points=None):
        """
        Called at the beginning of every iteration. The keyword arguments here are
        specified in modelfitting.initialize_criterion().
        """
        self.method=method
        if insets is not None:
            self.insets=array(append(insets,insets[-1]+100000000),dtype=int)
            self.outsets=array(append(outsets,outsets[-1]+100000000),dtype=int)
        if points is not None:
            self.points=array(append(points,points[-1]+100000000),dtype=int)
        self.next_index=0
        self.p = double(p)
        self.varname = varname

        self._error = zeros((self.K, self.N))
    
    def timestep_call(self):
        if self.method=='all':
            v = self.get_value(self.varname)
            t = self.step()+1
            if t<self.onset: return # onset
            if t*self.dt >= self.duration: return
            d = self.intdelays
            indices = (t-d>=0)&(t-d<self.total_steps) # neurons with valid delays (stay inside the target trace)
            vtar = self.traces[:,t-d] # target value at this timestep for every neuron
            for i in xrange(self.K):
                self._error[i,indices] += abs(v[indices]-vtar[i,indices])**self.p
                
        if self.method=='intervals':
            v = self.get_value(self.varname)
            t = self.step()+1
            if t<self.onset: return # onset
            if t*self.dt >= self.duration: return
            if t>=self.insets[self.next_index] and t<=self.outsets[self.next_index]:
                if t==self.outsets[self.next_index]:
                    self.next_index+=1
                d = self.intdelays
                indices = (t-d>=0)&(t-d<self.total_steps) # neurons with valid delays (stay inside the target trace)
                vtar = self.traces[:,t-d] # target value at this timestep for every neuron
                for i in xrange(self.K):
                    self._error[i,indices] += abs(v[indices]-vtar[i,indices])**self.p
                    
        if self.method=='points':
            v = self.get_value(self.varname)
            t = self.step()+1
            if t<self.onset: return # onset
            if t*self.dt >= self.duration: return
            if t==self.points[self.next_index]:
                
                self.next_index+=1
                d = self.intdelays
                indices = (t-d>=0)&(t-d<self.total_steps) # neurons with valid delays (stay inside the target trace)
                vtar = self.traces[:,t-d] # target value at this timestep for every neuron
                for i in xrange(self.K):
                    self._error[i,indices] += abs(v[indices]-vtar[i,indices])**self.p
                    
    def get_cuda_code(self):
        code = {}
        if self.method=='all':
            #  DECLARATION
            code['%CRITERION_DECLARE%'] = """
            double *error_arr,"""
            
            # INITIALIZATION
            code['%CRITERION_INIT%'] = """
            double error = error_arr[neuron_index];"""
            
            # TIMESTEP
            code['%CRITERION_TIMESTEP%'] = """
            if ((T >= onset)&(Tdelay<duration-spikedelay-1)) {
                error = error + pow(abs(trace_value - %s), %.4f);
            }
            """ % (self.varname, self.p)
            
            # FINALIZATION
            code['%CRITERION_END%'] = """
            error_arr[neuron_index] = error;"""
        
        if self.method=='intervals':
            #  DECLARATION
            code['%CRITERION_DECLARE%'] = """
            double *error_arr,
            int *insets,
            int *outsets,
            int *next_index_past,"""
            
            # INITIALIZATION
            code['%CRITERION_INIT%'] = """
            double error = error_arr[neuron_index];
            int next_index=*next_index_past;
            int inset=insets[next_index];
            int outset=outsets[next_index]; """
            
             #TIMESTEP
            code['%CRITERION_TIMESTEP%'] = """
           if ((T >= onset)&(Tdelay<duration-spikedelay-1)&(T >= inset)&(T <= outset)) {
                if (T == outset)
                    {next_index+=1;
                    inset=insets[next_index];
                    outset=outsets[next_index];
                    }
                 error = error + pow(abs(trace_value - %s), %.4f);
                 }
            
            """ %(self.varname, self.p)
      
            
            # FINALIZATION
            code['%CRITERION_END%'] = """
            error_arr[neuron_index] = error;
            *next_index_past=next_index;
            """
            
        if self.method=='points':
            #  DECLARATION
            code['%CRITERION_DECLARE%'] = """
            double *error_arr,
            int *points,
            int *next_index_past,"""
            
            # INITIALIZATION
            code['%CRITERION_INIT%'] = """
            double error = error_arr[neuron_index];
            int next_index=*next_index_past;
            int point=points[next_index]; """
            
            # TIMESTEP
            #(T >= onset)&(Tdelay<duration-spikedelay-1)&
            code['%CRITERION_TIMESTEP%'] = """
            if ((T==point)) {
                next_index+=1;
                point=points[next_index];
                error = error + pow(abs(trace_value - %s), %.4f);
            }
            """% (self.varname, self.p)
            
            
            # FINALIZATION
            code['%CRITERION_END%'] = """
            error_arr[neuron_index] = error;
            *next_index_past = next_index;
            """

        return code
    
    def initialize_cuda_variables(self):
        """
        Initialize GPU variables to pass to the kernel
        """
        self.error_gpu = gpuarray.to_gpu(zeros(self.N, dtype=double))
        if self.method=='intervals':
            self.insets_gpu = gpuarray.to_gpu(array(self.insets, dtype=int32))
            self.outsets_gpu = gpuarray.to_gpu(array(self.outsets, dtype=int32))
            self.next_index_past = gpuarray.to_gpu(array(int32(0)))
        if self.method=='points':
            self.points_gpu = gpuarray.to_gpu(array(self.points, dtype=int32))
            self.next_index_past = gpuarray.to_gpu(array(int32(0)))
            
    def get_kernel_arguments(self):
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        args = [self.error_gpu]
        if self.method=='intervals':
            args += [self.insets_gpu,self.outsets_gpu,self.next_index_past]
        if self.method=='points':
            args += [self.points_gpu,self.next_index_past]
        return args
    
    def update_gpu_values(self):
        """
        Call gpuarray.get() on final values, so that get_values() returns updated values.
        """
        self._error = self.error_gpu.get()
        if self.method=='intervals' or self.method=='points':
            self.index=self.next_index_past.get()
    
    def get_values(self):
        if self.K == 1: error = self._error.flatten()
        else: error = self._error
        #print self.index
        return error # just the integral, for every slice
    
    def normalize(self, error):
        # error is now the combined error on the whole duration (sum on the slices)
        # HACK: 1- because modelfitting MAXIMIZES for now...
#        self._norm = sum(self.traces**self.p, axis=1)**(1./self.p) # norm of every trace
        return 1-(self.dt*error)**(1./self.p)#/self._norm

class LpError(CriterionStruct):
    """
    Structure used by the users to specify a criterion
    """
    def __init__(self, p = 2, varname = 'v', method='all',insets=None,outsets=None,points=None):
        self.type = 'trace'
        self.p = p
        self.varname = varname
        self.method = method
        self.insets = insets
        self.outsets = outsets
        self.points=points


class GammaFactorCriterion2(Criterion):

    type = 'spikes'
    def initialize(self, delta=4*ms, fr_weight=0,coincidence_count_algorithm='exclusive',nlevels=1,level_duration=100):
        self.algorithm = coincidence_count_algorithm
        self.delta = int(rint(delta / self.dt))

        self.fr_weight = fr_weight
        self.spike_count = zeros(self.N, dtype='int')
        self.coincidences = zeros(self.N, dtype='int')
        self.spiketime_index = self.spikes_offset

        self.last_spike_time = array(rint(self.spikes_inline[self.spiketime_index] / self.dt), dtype=int)
        self.next_spike_time = array(rint(self.spikes_inline[self.spiketime_index + 1] / self.dt), dtype=int)
        self.ntrials = len(self.trials_offset)
        self.nlevels = nlevels
        self.level_duration=rint(level_duration/ self.dt)
        
        self.target_count=zeros(self.ntrials)
        self.target_rate=zeros(self.ntrials)
        ind_start=nonzero(self.spikes_inline==-1)[0]
        ind_start=append(ind_start,len(self.spikes_inline))
        self.trials_offset=tile(self.trials_offset,self.N)
        for itrial in xrange(self.ntrials):
            self.target_count[itrial]=ind_start[itrial+1]-ind_start[itrial]-2
            self.target_rate[itrial]=self.target_count[itrial]/self.spikes_inline[ind_start[itrial+1]-2]
        
        self.target_count_level=zeros((self.nlevels,self.ntrials))
        for ilevel in xrange(self.nlevels):
            for itrial in xrange(self.ntrials):
                temp=rint(self.spikes_inline[ind_start[itrial]+1:ind_start[itrial+1]-1]/self.dt)
#                print ilevel*self.level_duration,(ilevel+1)*self.level_duration,max(temp)
                ind=logical_and(temp>ilevel*self.level_duration,temp<=(ilevel+1)*self.level_duration)
#                print where(ind)[0]
                self.target_count_level[ilevel,itrial]=len(where(ind)[0])
        # First target spikes (needed for the computation of 
        #   the target train firing rates)
#        self.first_target_spike = zeros(self.N)

        self.last_spike_allowed = ones(self.N, dtype='bool')
        self.next_spike_allowed = ones(self.N, dtype='bool')
        
    
    if 1:
        #threadIdx.x+blockDim.x*itrial
        def get_cuda_code(self):
            block_size=32
            code = {}
            
            #  DECLARATION
            code['%CRITERION_DECLARE%'] = """
        int *spikecount,          // Number of spikes produced by each neuron
        int *num_coincidences,    // Count of coincidences for each neuron
            """
            
            if self.algorithm == 'exclusive':
                code['%CRITERION_DECLARE%'] += """ 
        int *sp_trial_indices,   // 
        int *last_spike_time_arr,  // pass zero array
        int *next_spike_time_arr,
        bool *last_spike_allowed_arr,
        bool *next_spike_allowed_arr,
        int *spike_count_level,
        int *current_level_index,
               """
            
            # INITIALIZATION
            code['%CRITERION_INIT%'] = """
         int ntrials=%i;//*ntrial;
         int nlevel=%i;
         int level_duration=%i;
        int level_ind = current_level_index[neuron_index];  
        int nspikes_level = spike_count_level[neuron_index*nlevel+level_ind];
        
        int nspikes = spikecount[neuron_index];   
        __shared__ int ncoinc[%i];
        __shared__ int sp_trial_indices_temp[%i];
        __shared__ int last_spike_time[%i];
        __shared__ int next_spike_time[%i];
        __shared__ bool last_spike_allowed[%i];
        __shared__ bool next_spike_allowed[%i];
    
       for(int itrial=0;itrial<ntrials;itrial++)
            {
            sp_trial_indices_temp[threadIdx.x*ntrials+itrial] = sp_trial_indices[neuron_index*ntrials+itrial];
            last_spike_time[threadIdx.x*ntrials+itrial] = last_spike_time_arr[neuron_index*ntrials+itrial];
            next_spike_time[threadIdx.x*ntrials+itrial] =next_spike_time_arr[neuron_index*ntrials+itrial];
            last_spike_allowed[threadIdx.x*ntrials+itrial] = last_spike_allowed_arr[neuron_index*ntrials+itrial];
            next_spike_allowed[threadIdx.x*ntrials+itrial] = next_spike_allowed_arr[neuron_index*ntrials+itrial];
            ncoinc[threadIdx.x*ntrials+itrial] = num_coincidences[neuron_index*ntrials+itrial];
            }  
       __syncthreads();
            """%(self.ntrials,self.nlevels,self.level_duration,self.ntrials*block_size,self.ntrials*block_size,self.ntrials*block_size,self.ntrials*block_size,self.ntrials*block_size,self.ntrials*block_size)
            
            
            # TIMESTEP
            code['%CRITERION_TIMESTEP%'] = """
            const int Tspike = T+spikedelay;
            """
            
            if self.algorithm == 'exclusive':
                code['%CRITERION_TIMESTEP%'] += """
           for(int itrial=0;itrial<ntrials;itrial++)
            {
                bool near_last_spike = last_spike_time[threadIdx.x*ntrials+itrial]+%d>=Tspike;
                bool near_next_spike = next_spike_time[threadIdx.x*ntrials+itrial]-%d<=Tspike;
                near_last_spike = near_last_spike && has_spiked;
                near_next_spike = near_next_spike && has_spiked;
    
                ncoinc[threadIdx.x*ntrials+itrial] +=(near_last_spike&&last_spike_allowed[threadIdx.x*ntrials+itrial]) || (near_next_spike&&next_spike_allowed[threadIdx.x*ntrials+itrial]);
                bool near_both_allowed = (near_last_spike&&last_spike_allowed[threadIdx.x*ntrials+itrial]) && (near_next_spike&&next_spike_allowed[threadIdx.x*ntrials+itrial]);
                last_spike_allowed[threadIdx.x*ntrials+itrial] = last_spike_allowed[threadIdx.x*ntrials+itrial] && !near_last_spike;
                next_spike_allowed[threadIdx.x*ntrials+itrial] = (next_spike_allowed[threadIdx.x*ntrials+itrial] && !near_next_spike) || near_both_allowed;
        
                if(Tspike>=next_spike_time[threadIdx.x*ntrials+itrial]){
                    sp_trial_indices_temp[threadIdx.x*ntrials+itrial]++;
                    last_spike_time[threadIdx.x*ntrials+itrial] = next_spike_time[threadIdx.x*ntrials+itrial];  
                    next_spike_time[threadIdx.x*ntrials+itrial] = spiketimes[sp_trial_indices_temp[threadIdx.x*ntrials+itrial]+1];
                }
                last_spike_allowed[threadIdx.x*ntrials+itrial] = next_spike_allowed[threadIdx.x*ntrials+itrial];
                next_spike_allowed[threadIdx.x*ntrials+itrial] = true;
     
            }

            nspikes += has_spiked*(T>=onset);
           if (T>(level_ind+1)*level_duration)
            {
            spike_count_level[neuron_index*nlevel+level_ind]=nspikes_level;
            level_ind+=1;
            nspikes_level=spike_count_level[neuron_index*nlevel+level_ind];
            }
            nspikes_level += has_spiked*(T>=onset);
           
            """% (self.delta, self.delta)
            
            # FINALIZATION
            code['%CRITERION_END%'] = """
            spike_count_level[neuron_index*nlevel+level_ind]=nspikes_level;
            if (Tend>(level_ind+1)*level_duration)
            {
            level_ind+=1;
            }
            
            for(int itrial=0;itrial<ntrials;itrial++)
                {
                sp_trial_indices[neuron_index*ntrials+itrial] = sp_trial_indices_temp[threadIdx.x*ntrials+itrial];
                num_coincidences[neuron_index*ntrials+itrial] = ncoinc[threadIdx.x*ntrials+itrial];
                last_spike_allowed_arr[neuron_index*ntrials+itrial] = last_spike_allowed[threadIdx.x*ntrials+itrial];
                next_spike_allowed_arr[neuron_index*ntrials+itrial] = next_spike_allowed[threadIdx.x*ntrials+itrial];
                last_spike_time_arr[neuron_index*ntrials+itrial]=last_spike_time[threadIdx.x*ntrials+itrial];
                next_spike_time_arr[neuron_index*ntrials+itrial]=next_spike_time[threadIdx.x*ntrials+itrial];
                }
            spikecount[neuron_index] = nspikes;
            current_level_index[neuron_index] = level_ind;
            __syncthreads();
                """
   
            
            return code
    else:
        def get_cuda_code(self):
            code = {}
            
            #  DECLARATION
            code['%CRITERION_DECLARE%'] = """
        int *spikecount,          // Number of spikes produced by each neuron
        int *num_coincidences,    // Count of coincidences for each neuron
            """
            
            if self.algorithm == 'exclusive':
                code['%CRITERION_DECLARE%'] += """ 
        int *sp_trial_indices,   // 
        int *last_spike_time_arr,  // pass zero array
        int *next_spike_time_arr,
        bool *last_spike_allowed_arr,
        bool *next_spike_allowed_arr,
               """
    
            
            # INITIALIZATION
            code['%CRITERION_INIT%'] = """
         int ntrials=%i;//*ntrial;
       int nspikes = spikecount[neuron_index];   
        
            """%(self.ntrials)
            
            # TIMESTEP
            code['%CRITERION_TIMESTEP%'] = """
            const int Tspike = T+spikedelay;
            """
            
            if self.algorithm == 'exclusive':
                code['%CRITERION_TIMESTEP%'] += """
           for(int itrial=0;itrial<ntrials;itrial++)
            {
                bool near_last_spike = last_spike_time_arr[neuron_index*ntrials+itrial]+%d>=Tspike;
                bool near_next_spike = next_spike_time_arr[neuron_index*ntrials+itrial]-%d<=Tspike;
                near_last_spike = near_last_spike && has_spiked;
                near_next_spike = near_next_spike && has_spiked;
    
                num_coincidences[neuron_index*ntrials+itrial] +=(near_last_spike&&last_spike_allowed_arr[neuron_index*ntrials+itrial]) || (near_next_spike&&next_spike_allowed_arr[neuron_index*ntrials+itrial]);
                bool near_both_allowed = (near_last_spike&&last_spike_allowed_arr[neuron_index*ntrials+itrial]) && (near_next_spike&&next_spike_allowed_arr[neuron_index*ntrials+itrial]);
                last_spike_allowed_arr[neuron_index*ntrials+itrial] = last_spike_allowed_arr[neuron_index*ntrials+itrial] && !near_last_spike;
                next_spike_allowed_arr[neuron_index*ntrials+itrial] = (next_spike_allowed_arr[neuron_index*ntrials+itrial] && !near_next_spike) || near_both_allowed;
        
                if(Tspike>=next_spike_time_arr[neuron_index*ntrials+itrial]){
                    sp_trial_indices[neuron_index*ntrials+itrial]+=1;
                    last_spike_time_arr[neuron_index*ntrials+itrial] = next_spike_time_arr[neuron_index*ntrials+itrial];  
                    next_spike_time_arr[neuron_index*ntrials+itrial] = spiketimes[sp_trial_indices[neuron_index*ntrials+itrial]+1];
                }
                last_spike_allowed_arr[neuron_index*ntrials+itrial] = next_spike_allowed_arr[neuron_index*ntrials+itrial];
                next_spike_allowed_arr[neuron_index*ntrials+itrial] = true;
     
            }
            nspikes += has_spiked*(T>=onset);
           
            """% (self.delta, self.delta)
            
            # FINALIZATION
            code['%CRITERION_END%'] = """
           spikecount[neuron_index] = nspikes;
            """
            
            return code
    
    def initialize_cuda_variables(self):
        """
        Initialize GPU variables to pass to the kernel
        """
        self.spike_count_gpu = gpuarray.to_gpu(zeros(self.N, dtype=int32))
        self.coincidences_gpu = gpuarray.to_gpu(zeros(self.N*self.ntrials, dtype=int32))
        self.sp_trial_indices_gpu = gpuarray.to_gpu(array(self.trials_offset,dtype=int32))
        self.last_spike_time_gpu = gpuarray.to_gpu(zeros(self.N*self.ntrials, dtype=int32))
        self.next_spike_time_gpu = gpuarray.to_gpu(zeros(self.N*self.ntrials, dtype=int32))
        self.last_spike_allowed_arr = gpuarray.to_gpu(zeros(self.N*self.ntrials, dtype=bool))
        self.next_spike_allowed_arr = gpuarray.to_gpu(ones(self.N*self.ntrials, dtype=bool))
        self.spike_count_level_gpu = gpuarray.to_gpu(zeros(self.N*self.nlevels, dtype=int32))
        self.current_level_index_gpu = gpuarray.to_gpu(zeros(self.N, dtype=int32))

    def get_kernel_arguments(self):
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        args = [self.spike_count_gpu, self.coincidences_gpu,self.sp_trial_indices_gpu,self.last_spike_time_gpu,self.next_spike_time_gpu]
        args += [self.last_spike_allowed_arr,self.next_spike_allowed_arr,self.spike_count_level_gpu,self.current_level_index_gpu]
        return args
    
    def update_gpu_values(self):
        """
        Call gpuarray.get() on final values, so that get_values() returns updated values.
        """
        self.coincidences = self.coincidences_gpu.get()
        self.spike_count = self.spike_count_gpu.get()
        self.spike_count_level = self.spike_count_level_gpu.get()
#        print self.sp_trial_indices_gpu.get()
    
    def get_values(self):
        return (self.coincidences, self.spike_count)
    
    def normalize(self, values):
        coincidence_count = values[0]
        spike_count = values[1]
        spike_count_temp = repeat(spike_count,self.ntrials)
#        self.target_count_temp = tile(self.target_count,self.N)
        delta = self.delta*self.dt

        if self.ntrials>1:
            coincidence_count = sum(reshape(coincidence_count,(self.N,self.ntrials)),axis=1)

        gamma = get_gamma_factor(coincidence_count, spike_count*self.ntrials, 
                                 sum(self.target_count), array(mean(self.target_rate)), 
                                 delta)
        self.spike_count_level=reshape(self.spike_count_level,(self.N,self.nlevels))
#        print self.target_count_level.shape,sum(self.target_count_level,axis=1)
#        print self.spike_count_level-sum(self.target_count_level,axis=1)
#        print abs(self.spike_count_level*self.ntrials-sum(self.target_count_level,axis=1))/sum(self.target_count_level,axis=1)
#        print self.target_count_level
        if self.fr_weight!=0:
            gamma = gamma - self.fr_weight*mean(abs(self.spike_count_level*self.ntrials-sum(self.target_count_level,axis=1))/sum(self.target_count_level,axis=1),axis=1)
        return gamma

class GammaFactor2(CriterionStruct):
    def __init__(self, delta = 4*ms,fr_weight=0, coincidence_count_algorithm = 'exclusive',nlevels=1,level_duration=100):
        self.type = 'spikes'
        self.delta = delta
        self.nlevels = nlevels
        self.fr_weight = fr_weight
        self.level_duration=level_duration
        self.coincidence_count_algorithm = coincidence_count_algorithm


class GammaFactorCriterion(Criterion):
    """
    Coincidence counter class.
    
    Counts the number of coincidences between the spikes of the neurons in the network (model spikes),
    and some user-specified data spike trains (target spikes). This number is defined as the number of 
    target spikes such that there is at least one model spike within +- ``delta``, where ``delta``
    is the half-width of the time window.
    
    Initialised as::
    
        cc = CoincidenceCounter(source, data, delta = 4*ms)
    
    with the following arguments:
    
    ``source``
        A :class:`NeuronGroup` object which neurons are being monitored.
    
    ``data``
        The list of spike times. Several spike trains can be passed in the following way.
        Define a single 1D array ``data`` which contains all the target spike times one after the
        other. Now define an array ``spiketimes_offset`` of integers so that neuron ``i`` should 
        be linked to target train: ``data[spiketimes_offset[i]], data[spiketimes_offset[i]+1]``, etc.
        
        It is essential that each spike train with the spiketimes array should begin with a spike at a
        large negative time (e.g. -1*second) and end with a spike that is a long time
        after the duration of the run (e.g. duration+1*second).
    
    ``delta=4*ms``
        The half-width of the time window for the coincidence counting algorithm.
    
    ``spiketimes_offset``
        A 1D array, ``spiketimes_offset[i]`` is the index of the first spike of 
        the target train associated to neuron i.
        
    ``spikedelays``
        A 1D array with spike delays for each neuron. All spikes from the target 
        train associated to neuron i are shifted by ``spikedelays[i]``.
        
    ``coincidence_count_algorithm``
        If set to ``'exclusive'``, the algorithm cannot count more than one
        coincidence for each model spike.
        If set to ``'inclusive'``, the algorithm can count several coincidences
        for a single model spike.
    
    ``onset``
        A scalar value in seconds giving the start of the counting: no
        coincidences are counted before ``onset``.
    
    Has three attributes:
    
    ``coincidences``
        The number of coincidences for each neuron of the :class:`NeuronGroup`.
        ``coincidences[i]`` is the number oflength[i]`` is the spike
  count for n i.
        
    ``target_length``
        The number of spikes in the target spike train associated to each neuron.
    """
    type = 'spikes'
    def initialize(self, delta=4*ms, fr_weight=0,coincidence_count_algorithm='exclusive'):
        self.algorithm = coincidence_count_algorithm
        self.delta = int(rint(delta / self.dt))
        self.fr_weight = fr_weight
        self.spike_count = zeros(self.N, dtype='int')
        self.coincidences = zeros(self.N, dtype='int')
        self.spiketime_index = self.spikes_offset
        self.last_spike_time = array(rint(self.spikes_inline[self.spiketime_index] / self.dt), dtype=int)
        self.next_spike_time = array(rint(self.spikes_inline[self.spiketime_index + 1] / self.dt), dtype=int)

        # First target spikes (needed for the computation of 
        #   the target train firing rates)
#        self.first_target_spike = zeros(self.N)

        self.last_spike_allowed = ones(self.N, dtype='bool')
        self.next_spike_allowed = ones(self.N, dtype='bool')
        
    def spike_call(self, spiking_neurons):
        dt = self.dt
        t = self.step()*dt
        
        spiking_neurons = array(spiking_neurons)
        if len(spiking_neurons)>0:
            
            if t >= self.onset:
                self.spike_count[spiking_neurons] += 1

            T_spiking = array(rint((t + self.delays[spiking_neurons]) / dt), dtype=int)

            remaining_neurons = spiking_neurons
            remaining_T_spiking = T_spiking
            while True:
                remaining_indices, = (remaining_T_spiking > self.next_spike_time[remaining_neurons]).nonzero()
                if len(remaining_indices):
                    indices = remaining_neurons[remaining_indices]
                    self.spiketime_index[indices] += 1
                    self.last_spike_time[indices] = self.next_spike_time[indices]
                    self.next_spike_time[indices] = array(rint(self.spikes_inline[self.spiketime_index[indices] + 1] / dt), dtype=int)
                    if self.algorithm == 'exclusive':
                        self.last_spike_allowed[indices] = self.next_spike_allowed[indices]
                        self.next_spike_allowed[indices] = True
                    remaining_neurons = remaining_neurons[remaining_indices]
                    remaining_T_spiking = remaining_T_spiking[remaining_indices]
                else:
                    break

            # Updates coincidences count
            near_last_spike = self.last_spike_time[spiking_neurons] + self.delta >= T_spiking
            near_next_spike = self.next_spike_time[spiking_neurons] - self.delta <= T_spiking
            last_spike_allowed = self.last_spike_allowed[spiking_neurons]
            next_spike_allowed = self.next_spike_allowed[spiking_neurons]
            I = (near_last_spike & last_spike_allowed) | (near_next_spike & next_spike_allowed)

            if t >= self.onset:
                self.coincidences[spiking_neurons[I]] += 1

            if self.algorithm == 'exclusive':
                near_both_allowed = (near_last_spike & last_spike_allowed) & (near_next_spike & next_spike_allowed)
                self.last_spike_allowed[spiking_neurons] = last_spike_allowed & -near_last_spike
                self.next_spike_allowed[spiking_neurons] = (next_spike_allowed & -near_next_spike) | near_both_allowed
    
    def get_cuda_code(self):
        code = {}
        
        #  DECLARATION
        code['%CRITERION_DECLARE%'] = """
    int *spikecount,          // Number of spikes produced by each neuron
    int *num_coincidences,    // Count of coincidences for each neuron
        """
        
        if self.algorithm == 'exclusive':
            code['%CRITERION_DECLARE%'] += """ 
    bool *last_spike_allowed_arr,
    bool *next_spike_allowed_arr,
           """
        
        # INITIALIZATION
        code['%CRITERION_INIT%'] = """
    int ncoinc = num_coincidences[neuron_index];
    int nspikes = spikecount[neuron_index];
    int last_spike_time = spiketimes[spiketime_index];
    int next_spike_time = spiketimes[spiketime_index+1];
        """
        
        if self.algorithm == 'exclusive':
            code['%CRITERION_INIT%'] += """
    bool last_spike_allowed = last_spike_allowed_arr[neuron_index];
    bool next_spike_allowed = next_spike_allowed_arr[neuron_index];
            """
        
        # TIMESTEP
        code['%CRITERION_TIMESTEP%'] = """
        const int Tspike = T+spikedelay;
        """
        
        if self.algorithm == 'inclusive':
            code['%CRITERION_TIMESTEP%'] += """ 
    ncoinc += has_spiked && (((last_spike_time+%d)>=Tspike) || ((next_spike_time-%d)<=Tspike));
            """ % (self.delta, self.delta)
        if self.algorithm == 'exclusive':
            code['%CRITERION_TIMESTEP%'] += """
        bool near_last_spike = last_spike_time+%d>=Tspike;
        bool near_next_spike = next_spike_time-%d<=Tspike;
        near_last_spike = near_last_spike && has_spiked;
        near_next_spike = near_next_spike && has_spiked;
        ncoinc += (near_last_spike&&last_spike_allowed) || (near_next_spike&&next_spike_allowed);
        bool near_both_allowed = (near_last_spike&&last_spike_allowed) && (near_next_spike&&next_spike_allowed);
        last_spike_allowed = last_spike_allowed && !near_last_spike;
        next_spike_allowed = (next_spike_allowed && !near_next_spike) || near_both_allowed;
            """ % (self.delta, self.delta)
    
        code['%CRITERION_TIMESTEP%'] += """
        nspikes += has_spiked*(T>=onset);
        if(Tspike>=next_spike_time){
            spiketime_index++;
            last_spike_time = next_spike_time;
            next_spike_time = spiketimes[spiketime_index+1];
        """
    
        if self.algorithm == 'exclusive':
            code['%CRITERION_TIMESTEP%'] += """
        last_spike_allowed = next_spike_allowed;
        next_spike_allowed = true;
            """
    
        code['%CRITERION_TIMESTEP%'] += """
        }
        """
        
        # FINALIZATION
        code['%CRITERION_END%'] = """
    num_coincidences[neuron_index] = ncoinc;
    spikecount[neuron_index] = nspikes;
        """
        
        if self.algorithm == 'exclusive':
            code['%CRITERION_END%'] += """
    last_spike_allowed_arr[neuron_index] = last_spike_allowed;
    next_spike_allowed_arr[neuron_index] = next_spike_allowed;
            """
        
        return code
    
    def initialize_cuda_variables(self):
        """
        Initialize GPU variables to pass to the kernel
        """
        self.spike_count_gpu = gpuarray.to_gpu(zeros(self.N, dtype=int32))
        self.coincidences_gpu = gpuarray.to_gpu(zeros(self.N, dtype=int32))
        
        if self.algorithm == 'exclusive':
            self.last_spike_allowed_arr = gpuarray.to_gpu(zeros(self.N, dtype=bool))
            self.next_spike_allowed_arr = gpuarray.to_gpu(ones(self.N, dtype=bool))
    
    def get_kernel_arguments(self):
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        args = [self.spike_count_gpu, self.coincidences_gpu]
        
        if self.algorithm == 'exclusive':
            args += [self.last_spike_allowed_arr,
                     self.next_spike_allowed_arr]
        return args
    
    def update_gpu_values(self):
        """
        Call gpuarray.get() on final values, so that get_values() returns updated values.
        """
        self.coincidences = self.coincidences_gpu.get()
        self.spike_count = self.spike_count_gpu.get()
    
    def get_values(self):
        return (self.coincidences, self.spike_count)
    
    def normalize(self, values):
        coincidence_count = values[0]
        spike_count = values[1]
        delta = self.delta*self.dt

        gamma = get_gamma_factor(coincidence_count, spike_count, 
                                 self.target_spikes_count, self.target_spikes_rate, 
                                 delta)

        gamma = gamma - self.fr_weight*abs(spike_count-self.target_spikes_count)/self.target_spikes_count
#        print 'fr',abs(spike_count-self.target_spikes_count)/self.target_spikes_count,'fr'
        return gamma

class GammaFactor(CriterionStruct):
    def __init__(self, delta = 4*ms,fr_weight=0, coincidence_count_algorithm = 'exclusive'):
        self.type = 'spikes'
        self.delta = delta
        self.fr_weight = fr_weight
        self.coincidence_count_algorithm = coincidence_count_algorithm




class VanRossumCriterion(Criterion):
    type = 'spikes'
    def initialize(self, tau):
        self.delay_range =max(self.delays)- min(self.delays)#delay range
        self.min_delay = abs(min(self.delays))#minimum of possible delay
        self.distance_vector=zeros(self.N) 
        self.nbr_neurons_group = self.N/self.K
        
        eqs="""
        dv/dt=(-v)/tau: volt
        """
        # network to convolve target spikes with the kernel
        self.input_target=SpikeGeneratorGroup(self.K,self.spikes,clock=self.group.clock)
        self.kernel_target=NeuronGroup(self.K,model=eqs,clock=self.group.clock)
        self.C_target = DelayConnection(self.input_target, self.kernel_target, 'v', structure='dense',  max_delay=self.min_delay)     
        self.C_target.connect_one_to_one(self.input_target,self.kernel_target)
        self.C_target.delay = self.min_delay*ones_like(self.C_target.delay)

        # network to convolve population spikes with the kernel
        self.kernel_population=NeuronGroup(self.N,model=eqs,clock=self.group.clock)
        self.C_population = DelayConnection(self.group, self.kernel_population, 'v', structure='sparse',  max_delay=self.delay_range)
        for iN in xrange(self.N):
            self.C_population.delay[iN,iN] = diagflat(self.min_delay + self.delays[iN])
        self.C_population.connect_one_to_one(self.group,self.kernel_population)
        self.spikecount_mon = SpikeCounter(self.group)
        self.contained_objects = [self.kernel_population,self.C_population,self.spikecount_mon,self.input_target,self.C_target,self.kernel_target]  
        self.dt=self.group.clock.dt
        self.tau = tau

    def timestep_call(self):
        trace_population = self.kernel_population.state_('v')
        trace_target = self.kernel_target.state_('v')
        for igroup in xrange(self.K):
            self.distance_vector[igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group] += (trace_population[igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group]-trace_target[igroup])**2
        self.spikecount=self.spikecount_mon.count
    def get_values(self):
        return (self.distance_vector)
    
    def normalize(self, distance_vector):
        distance_vector[nonzero(self.spikecount==0)] = inf
        #distance_vector[nonzero(self.spikecount==1)] = inf
        return 1-distance_vector*self.group.clock.dt

    def get_cuda_code(self):
        code = {}
        
        #  DECLARATION
        code['%CRITERION_DECLARE%'] = """
    double *error_arr,
    int *count_arr,
        """
        
        # INITIALIZATION
        code['%CRITERION_INIT%'] = """
    double error = error_arr[neuron_index];
    double V_kernel =0;
    double V_target =0;
    int count=count_arr[neuron_index];
    int next_spike_time = spiketimes[spiketime_index+1];
        """
        
        # TIMESTEP
#        code['%CRITERION_TIMESTEP%'] = """
#        """
        code['%CRITERION_TIMESTEP%'] = """
        const int Tspike = T+spikedelay;
        
        if(has_spiked){       // If the neuron has spiked
            V_kernel+=1;
        }
        
        count += has_spiked*(T>=onset);
        if(Tspike>=next_spike_time){
            V_target+=1;
            spiketime_index++;
            next_spike_time = spiketimes[spiketime_index+1]; }
            

        V_kernel=V_kernel*exp(-%.8f/%.6f);
        V_target=V_target*exp(-%.8f/%.6f);
        error+=pow(V_kernel-V_target,2);
        
        
        """ % (self.dt,self.tau,self.dt,self.tau)
        
        # FINALIZATION
        code['%CRITERION_END%'] = """
    error_arr[neuron_index] = error;
    count_arr[neuron_index] = count;
        """
        
        return code
    
    def initialize_cuda_variables(self):
        """
        Initialize GPU variables to pass to the kernel
        """
        self.error_gpu = gpuarray.to_gpu(zeros(self.N, dtype=double))
        self.count_gpu = gpuarray.to_gpu(zeros(self.N, dtype=int32))
    
    def get_kernel_arguments(self):
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        args = [self.error_gpu,self.count_gpu]
        return args
    
    def update_gpu_values(self):
        """
        Call gpuarray.get() on final values, so that get_values() returns updated values.
        """
        self.distance_vector = self.error_gpu.get()
        self.spikecount = self.count_gpu.get()

        
class VanRossum(CriterionStruct):
    def __init__(self, tau):
        self.type = 'spikes'
        self.tau = tau


class BretteCriterion(Criterion):
    type = 'spikes'
    def initialize(self,tau_metric):
#        tau_metric=tau_metric**
        self.delay_range =max(self.delays)- min(self.delays)#delay range
        self.min_delay = abs(min(self.delays))#minimum of possible delay
        self.corr_vector=zeros(self.N) 
        self.norm_pop = zeros(self.N) 
        self.norm_target = zeros(self.N) 
        self.nbr_neurons_group = self.N/self.K
        
        eqs="""
        tau:second
        dv/dt=(-v)/tau: volt
        """
        # network to convolve target spikes with the kernel
        self.input_target=SpikeGeneratorGroup(self.K,self.spikes,clock=self.group.clock)
        self.kernel_target=NeuronGroup(self.N,model=eqs,clock=self.group.clock)
        self.C_target = DelayConnection(self.input_target, self.kernel_target, 'v', structure='sparse',  max_delay=self.min_delay)  
        self.kernel_target.tau=tau_metric

        for igroup in xrange(self.K):
            self.C_target.W[igroup,igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group] = ones(self.nbr_neurons_group)
            self.C_target.delay[igroup,igroup*self.nbr_neurons_group:(1+igroup)*self.nbr_neurons_group] =  self.min_delay * ones(self.nbr_neurons_group)
            
        # network to convolve population spikes with the kernel
        self.kernel_population=NeuronGroup(self.N,model=eqs,clock=self.group.clock)
        self.C_population = DelayConnection(self.group, self.kernel_population, 'v', structure='sparse',  max_delay=self.delay_range)
        for iN in xrange(self.N):
            self.C_population.delay[iN,iN] = diagflat(self.min_delay + self.delays[iN])
        self.C_population.connect_one_to_one(self.group,self.kernel_population)
        self.kernel_population.tau=tau_metric
        self.spikecount_mon = SpikeCounter(self.group)
        self.contained_objects = [self.kernel_population,self.C_population,self.spikecount_mon,self.input_target,self.C_target,self.kernel_target]  
        self.tau_metric=tau_metric
        self.dt=self.group.clock.dt
        
    def timestep_call(self):
        trace_population = self.kernel_population.state_('v')
        trace_target = self.kernel_target.state_('v')
        self.corr_vector += trace_population*trace_target
        self.norm_pop += trace_population**2
        self.norm_target += trace_target**2
        self.spikecount=self.spikecount_mon.count
        
    def get_values(self):
        #print self.corr_vector,self.norm_pop,self.norm_target
        return (self.corr_vector,self.norm_pop,self.norm_target)
    
    def normalize(self, values):
        corr_vector=values[0]
        norm_pop=values[1]
        norm_target=values[2]
        corr_vector[nonzero(self.spikecount==0)] = -inf
        temp=self.corr_vector/self.tau_metric/sqrt(norm_pop)/sqrt(norm_target)
        temp[isnan(temp)] = -inf
        return temp

    def get_cuda_code(self):
        code = {}
        
        #  DECLARATION
        code['%CRITERION_DECLARE%'] = """
    double *error_arr,
    int *count_arr,
    double *tau_metric_arr,
    double *norm_pop_arr,
    double *norm_target_arr, 
        """

        
        # INITIALIZATION
        code['%CRITERION_INIT%'] = """
    double error = error_arr[neuron_index];
    double V_kernel =0;
    double V_target =0;
    double tau_metric=tau_metric_arr[neuron_index];
    int count=count_arr[neuron_index];
    int next_spike_time = spiketimes[spiketime_index+1];
    double norm_pop=norm_pop_arr[neuron_index];
    double norm_target=norm_target_arr[neuron_index];
        """
        
        # TIMESTEP
#        code['%CRITERION_TIMESTEP%'] = """
#        """
        code['%CRITERION_TIMESTEP%'] = """
        const int Tspike = T+spikedelay;
        
        if(has_spiked){       // If the neuron has spiked
            V_kernel+=1;
        }
        
        count += has_spiked*(T>=onset);
        if(Tspike>=next_spike_time){
            V_target+=1;
            spiketime_index++;
            next_spike_time = spiketimes[spiketime_index+1]; }
        
        V_kernel=V_kernel*exp(-%.8f/tau_metric);
        V_target=V_target*exp(-%.8f/tau_metric);
        
        error+=V_kernel*V_target;
        norm_pop+=pow(V_kernel,2);
        norm_target +=pow(V_target,2);
        
        
        """ % (self.dt,self.dt)
        
        # FINALIZATION
        code['%CRITERION_END%'] = """
    error_arr[neuron_index] = error;
    count_arr[neuron_index] = count;
    norm_pop_arr[neuron_index] = norm_pop;
    norm_target_arr[neuron_index] = norm_target;
        """
        
        return code
    
    def initialize_cuda_variables(self):
        """
        Initialize GPU variables to pass to the kernel
        """
        self.error_gpu = gpuarray.to_gpu(zeros(self.N, dtype=double))
        self.count_gpu = gpuarray.to_gpu(zeros(self.N, dtype=int32))
        self.tau_metric_gpu = gpuarray.to_gpu(array(self.tau_metric, dtype=double))
        self.norm_pop_gpu = gpuarray.to_gpu(zeros(self.N, dtype=double))
        self.norm_target_gpu = gpuarray.to_gpu(zeros(self.N, dtype=double))
        
    def get_kernel_arguments(self):
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        args = [self.error_gpu,self.count_gpu,self.tau_metric_gpu,self.norm_pop_gpu,self.norm_target_gpu]
        return args
    
    def update_gpu_values(self):
        """
        Call gpuarray.get() on final values, so that get_values() returns updated values.
        """
        self.corr_vector = self.error_gpu.get()
        self.spikecount = self.count_gpu.get()
        self.norm_pop = self.norm_pop_gpu.get()
        self.norm_target = self.norm_target_gpu.get()

class Brette(CriterionStruct):
    def __init__(self):
        self.type = 'spikes'
