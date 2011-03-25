
if criterion_name == 'LpError2':
    params['p'] = self.criterion.p
    params['varname'] = self.criterion.varname
    params['insets'] = self.criterion.insets
    params['outsets'] = self.criterion.outsets
    self.criterion_object = LpErrorCriterion2(**params)





class LpErrorCriterion2(Criterion):
    type = 'traces'
    def initialize(self, p=2, varname='v',insets=None,outsets=None):
        """
        Called at the beginning of every iteration. The keyword arguments here are
        specified in modelfitting.initialize_criterion().
        """
        self.p = double(p)
        self.varname = varname
        self._error = zeros((self.K, self.N))
        self.insets = insets
        self.outsets = outsets
        self.next_index=0
        self.insets = array(append(insets,inf),dtype=int)
        self.outsets =  array(append(outsets,inf),dtype=int)
        
    def timestep_call(self):
        v = self.get_value(self.varname)
        t = self.step()+1
        if t<self.onset: return # onset
        if t*self.dt >= self.duration: return
        if t>=self.insets[self.next_index]and t<=self.outsets[self.next_index]:
            if t==self.outsets[self.next_index]:
                self.next_index+=1
            d = self.intdelays
            indices = (t-d>=0)&(t-d<self.total_steps) # neurons with valid delays (stay inside the target trace)
            vtar = self.traces[:,t-d] # target value at this timestep for every neuron
            for i in xrange(self.K):
                self._error[i,indices] += abs(v[indices]-vtar[i,indices])**self.p
    
    def get_cuda_code(self):
        code = {}
        #    double p,
        #  DECLARATION
        code['%CRITERION_DECLARE%'] = """
    double *error_arr,
    double *insets,
    double *outsets,

        """
        
        # INITIALIZATION
        code['%CRITERION_INIT%'] = """
    double error = error_arr[neuron_index];
    int next_index=0;
        """
        
        # TIMESTEP
        #)
        code['%CRITERION_TIMESTEP%'] = """
        if ((T >= onset)&(Tdelay<duration-spikedelay-1)&(T >= insets[next_index])&(T <= outsets[next_index])) {
            if (T == outsets[next_index])
                {next_index+=1;}
            error = error + pow(abs(trace_value - %s), %.4f);
        }
        """ % (self.varname, self.p)
        
        # FINALIZATION
        code['%CRITERION_END%'] = """
    error_arr[neuron_index] = error;
        """
        
        return code
    
    def initialize_cuda_variables(self):
        """
        Initialize GPU variables to pass to the kernel
        """
        self.error_gpu = gpuarray.to_gpu(zeros(self.N, dtype=double))
        self.insets_gpu = gpuarray.to_gpu(self.insets)
        self.outsets_gpu = gpuarray.to_gpu(self.outsets)
    
    def get_kernel_arguments(self):
        """
        Return a list of objects to pass to the CUDA kernel.
        """
        args = [self.error_gpu,self.insets_gpu,self.outsets_gpu]
        return args
    
    def update_gpu_values(self):
        """
        Call gpuarray.get() on final values, so that get_values() returns updated values.
        """
        self._error = self.error_gpu.get()
        print self._error
    
    def get_values(self):
        if self.K == 1: error = self._error.flatten()
        else: error = self._error
        return error # just the integral, for every slice
    
    def normalize(self, error):
        # error is now the combined error on the whole duration (sum on the slices)
        # HACK: 1- because modelfitting MAXIMIZES for now...
#        self._norm = sum(self.traces**self.p, axis=1)**(1./self.p) # norm of every trace
        return 1-(self.dt*error)**(1./self.p)#/self._norm

class LpError2(CriterionStruct):
    """
    Structure used by the users to specify a criterion
    """
    def __init__(self, p = 2, varname = 'v',insets=None,outsets=None):
        self.type = 'trace'
        self.p = p
        self.insets = insets
        self.outsets = outsets
        self.varname = varname
