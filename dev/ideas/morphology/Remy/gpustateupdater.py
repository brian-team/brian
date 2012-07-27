import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from brian.stateupdater import NonlinearStateUpdater
from numpy import zeros, ones
import numpy
import brian.optimiser as optimiser
from brian.inspection import get_identifiers
from time import time


class GPUExponentialEulerStateUpdater(NonlinearStateUpdater):
    def __init__(self, eqs, clock=None, freeze=False):
        '''
        Initialize a nonlinear model with dynamics dX/dt = f(X).
        '''
        # TODO: global pref?
        self.eqs = eqs
        self.optimized = compile
        self.changed = True
        self.prepared=False
        self.firstTime = True
        self.index_to_varname = []
        self.varname_to_index = {}
        self.index_nonzero = []
        if freeze:
            self.eqs.compile_functions(freeze=freeze)
        self._frozen=freeze
        
        self.compile_functions_C(eqs)
        
        mod = SourceModule("""
        __global__ void integrate(double *A,double *B,double *S_in,double *S_out,double dt)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          double a = A[idx];
          double b = B[idx];
          S_out[idx]= - b + (S_in[idx] + b) * (1.+a*dt*(1.+.5*a*dt*(1.+(1./3.)*a*dt)));
        }
        
        __global__ void copyVar(double *S_new, double *old, int var_index, int var_len)
        { 
          int idx =  var_index + blockIdx.x * var_len;
          S_new[idx]= old[idx];
        }
        
        __global__ void getV(double *S_out, double *v, int var_index, int var_len)
        { 
          int idx =  var_index + blockIdx.x * var_len;
          S_out[idx] = v[blockIdx.x];
        }
        
        __global__ void find_variation(double *S_in,double *S_out,double variation,double dt)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          double delta = fabs(S_out[idx] - S_in[idx])/dt;
          double onepercent = 0.01 * fabs(S_in[idx]);
          if(delta > onepercent){
              variation = 1.1;
          }
        }
        """)
        var_len  = len(dict.fromkeys(self.eqs._diffeq_names))+1 
        self.func = mod.get_function("integrate")
        self.func.prepare(["P","P","P","P","d"],block=(var_len,1,1))
        
        self.copyVar = mod.get_function("copyVar")
        self.copyVar.prepare(["P","P","i","i"],block=(1,1,1))
        
        self.getV = mod.get_function("getV")#idea : if neuron did not change : use this to update S_out and then copy S_out to S_in
        self.getV.prepare(["P","P","i","i"],block=(1,1,1))
        
        self.find_var = mod.get_function("find_variation")
        self.find_var.prepare(["P","P",'d',"d"],block=(var_len,1,1))
    
    def prepare(self, P):
        n = len(P.state_(self.eqs._diffeq_names_nonzero[0]))
        var_len  = len(dict.fromkeys(self.eqs._diffeq_names))+1 # +1 needed to store t
        
        for index,varname in enumerate(self.eqs._diffeq_names):
            self.index_to_varname.append(varname)
            self.varname_to_index[varname]= index
            if varname in self.eqs._diffeq_names_nonzero :
                self.index_nonzero.append(index)
        
        self.S_in = cuda.pagelocked_zeros((n,var_len),numpy.float64)
        
        self.S_out = cuda.pagelocked_zeros((n,var_len),numpy.float64)
        
        nbytes = n * var_len * numpy.dtype(numpy.float64).itemsize
        self.S_in_gpu = cuda.mem_alloc(nbytes)
        self.S_out_gpu = cuda.mem_alloc(nbytes)
        
        Z = zeros((n,var_len))
        self.A_gpu = cuda.mem_alloc(nbytes)
        cuda.memcpy_htod(self.A_gpu, Z)
        self.B_gpu = cuda.mem_alloc(nbytes)
        cuda.memcpy_htod(self.B_gpu, Z)
        self.S_temp_gpu = cuda.mem_alloc(nbytes)
        
        modFun={}
        self.applyFun = {}
        for x in self.index_nonzero:
            s = self.eqs._function_C_String[self.index_to_varname[x]]
            args_fun =[]
            for i in xrange(var_len):
                args_fun.append("S_temp["+str(i)+" + blockIdx.x * var_len]")
            modFun[x] = SourceModule("""
                __device__ double f"""+ s +"""
                
                __global__ void applyFun(double *A,double *B,double *S_in,double *S_temp, int x, int var_len)
                { 
                    
                    int idx = x + blockIdx.x * var_len;
                    S_temp[idx] = 0;
                    B[idx] = f("""+",".join(args_fun)+""");
                    S_temp[idx] = 1;
                    A[idx] = f("""+",".join(args_fun)+""") - B[idx];
                    B[idx] /= A[idx];
                    S_temp[idx] = S_in[idx];
                }
                """)
            self.applyFun[x] = modFun[x].get_function("applyFun")
            self.applyFun[x].prepare(['P','P','P','P','i','i'],block=(1,1,1))
        
        self.calc_dict = {}
        self.already_calc = {}
    
    def __call__(self, P):
        
        if self.prepared==False:
            self.prepare(P)
            self.prepared = True
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        '''
        var_len  = len(dict.fromkeys(self.eqs._diffeq_names))+1 # +1 needed to store t
        
        if self.changed :
            if not self.firstTime:
                cuda.memcpy_dtoh(self.S_out, self.S_out_gpu)
                for var in self.eqs._diffeq_names_nonzero:
                    Temp = P.state_(var)
                    Temp[:] = self.S_out[:,self.varname_to_index[var]]
            else :
                self.firstTime = False
            
            for var in self.eqs._diffeq_names:
                self.S_in[:,self.varname_to_index[var]] = P.state_(var)
            if self._frozen:
                self.S_in[:,var_len-1]=P.clock._t # without units
            else:
                self.S_in[:,var_len-1] = P.clock.t #time
            cuda.memcpy_htod(self.S_in_gpu, self.S_in)
            cuda.memcpy_dtod(self.S_out_gpu,self.S_in_gpu, self.S_in.size * self.S_in.dtype.itemsize)
            cuda.memcpy_dtod(self.S_temp_gpu,self.S_in_gpu, self.S_in.size * self.S_in.dtype.itemsize)
        else :
            cuda.memcpy_dtod(self.S_in_gpu,self.S_out_gpu, self.S_in.size * self.S_in.dtype.itemsize)
            cuda.memcpy_dtod(self.S_temp_gpu,self.S_out_gpu, self.S_in.size * self.S_in.dtype.itemsize)
          
        self.gpu_exponential_euler(P.clock._dt)
        self.changed = False
    
    def updateV(self,v_gpu):
        n = len(self.S_in)
        var_len = len(self.S_in[0])
        var_lenBis = numpy.int32(var_len)
        vx = numpy.int32(self.varname_to_index['v'])
        self.getV.prepared_call((n,1),self.S_out_gpu,v_gpu,vx,var_lenBis)
    
    def gpu_exponential_euler(self,dt):
        n = len(self.S_in)
        var_len = len(self.S_in[0])
        var_lenBis = numpy.int32(var_len)
         
        for x in self.index_nonzero:
            x = numpy.int32(x)
            self.applyFun[x].prepared_call((n,1),self.A_gpu, self.B_gpu, self.S_in_gpu, self.S_temp_gpu, x, var_lenBis)    
        
        dt = numpy.float64(dt)
        
        self.func.prepared_call((n,1),self.A_gpu,self.B_gpu,self.S_in_gpu,self.S_out_gpu,dt)
        
    def calc(self,var,res_gpu):
        n = len(self.S_in)
        var_len = len(self.S_in[0])
        
        if var not in self.already_calc:
            vars_strings = []
            for var_aux in self.eqs._diffeq_names:
                vars_strings.append("double "+ var_aux)
            vars_strings.append("double t")
            
            expr = self.eqs._string[var]
            namespace = self.eqs._namespace[var] # name space of the function
            all_variables = self.eqs._eq_names + self.eqs._diffeq_names + self.eqs._alias.keys() + ['t']
            expr = optimiser.freeze(expr, all_variables, namespace)
            s = "(" + ",".join(vars_strings) + ") { double result = " + expr + "; return result; }"
            for var_aux in self.eqs._diffeq_names: #this is ugly. really ugly.
                s = s.replace(var_aux+'**2',var_aux+'*'+var_aux)
                s = s.replace(var_aux+'**3',var_aux+'*'+var_aux+'*'+var_aux)
                s = s.replace(var_aux+'**4',var_aux+'*'+var_aux+'*'+var_aux+'*'+var_aux)
                s = s.replace(var_aux+'**5',var_aux+'*'+var_aux+'*'+var_aux+'*'+var_aux+'*'+var_aux)
            args_fun =[]
            for i in xrange(var_len):
                args_fun.append("S_out["+str(i)+" + blockIdx.x * var_len]")
            mod = SourceModule("""
                    __device__ double f"""+ s +"""
                    
                    __global__ void calc(double *res,double *S_out, int var_len)
                    { 
                        int idx = blockIdx.x;
                        res[idx] = f("""+",".join(args_fun)+""");
                        
                    }
                    """)
            self.calc_dict[var] = mod.get_function("calc")
            self.calc_dict[var].prepare(['P','P','i'],block=(1,1,1))
            self.already_calc[var] = True
        self.calc_dict[var].prepared_call((n,1),res_gpu,self.S_out_gpu,numpy.int32(var_len))
    
    def compile_functions_C(self, eqs, freeze=False):
        """
        Compile all functions defined as strings.
        If freeze is True, all external parameters and units are replaced by their value.
        ALL FUNCTIONS MUST HAVE STRINGS.
        """
        all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
        # Check if freezable
        freeze = freeze and all([optimiser.freeze(expr, all_variables, eqs._namespace[name])\
                               for name, expr in eqs._string.iteritems()])
        eqs._frozen = freeze
        
        vars_strings = []
        for var in eqs._diffeq_names:
            vars_strings.append("double "+ var)
        vars_strings.append("double t")
        # Compile strings to functions
        for name, expr in eqs._string.iteritems():
            namespace = eqs._namespace[name] # name space of the function
            expr = optimiser.freeze(expr, all_variables, namespace)
            s = "(" + ",".join(vars_strings) + ") { double result = " + expr + "; return result; }"
            eqs._function_C_String[name] = s
            
class GPUExponentialEulerStateUpdater_float(NonlinearStateUpdater):
    def __init__(self, eqs, clock=None, freeze=False):
        '''
        Initialize a nonlinear model with dynamics dX/dt = f(X).
        '''
        # TODO: global pref?
        self.eqs = eqs
        self.optimized = compile
        self.changed = True
        self.prepared=False
        self.firstTime = True
        self.index_to_varname = []
        self.varname_to_index = {}
        self.index_nonzero = []
        if freeze:
            self.eqs.compile_functions(freeze=freeze)
        self._frozen=freeze
        
        self.compile_functions_C(eqs)
        
        mod = SourceModule("""
        __global__ void integrate(float *A,float *B,float *S_in,float *S_out,float dt)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          float a = A[idx];
          float b = B[idx];
          S_out[idx]= - b + (S_in[idx] + b) * (1.+a*dt*(1.+.5*a*dt*(1.+(1./3.)*a*dt)));
        }
        
        __global__ void copyVar(float *S_new, float *old, int var_index, int var_len)
        { 
          int idx =  var_index + blockIdx.x * var_len;
          S_new[idx]= old[idx];
        }
        
        __global__ void getV(float *S_out, float *v, int var_index, int var_len)
        { 
          int idx =  var_index + blockIdx.x * var_len;
          S_out[idx] = v[blockIdx.x];
        }
        
        __global__ void find_variation(float *S_in,float *S_out,float variation,float dt)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          float delta = fabs(S_out[idx] - S_in[idx])/dt;
          float onepercent = 0.01 * fabs(S_in[idx]);
          if(delta > onepercent){
              variation = 1.1;
          }
        }
        """)
        var_len  = len(dict.fromkeys(self.eqs._diffeq_names))+1 
        self.func = mod.get_function("integrate")
        self.func.prepare(["P","P","P","P","f"],block=(var_len,1,1))
        
        self.copyVar = mod.get_function("copyVar")
        self.copyVar.prepare(["P","P","i","i"],block=(1,1,1))
        
        self.getV = mod.get_function("getV")#idea : if neuron did not change : use this to update S_out and then copy S_out to S_in
        self.getV.prepare(["P","P","i","i"],block=(1,1,1))
        
        self.find_var = mod.get_function("find_variation")
        self.find_var.prepare(["P","P",'f',"f"],block=(var_len,1,1))
    
    def prepare(self, P):
        n = len(P.state_(self.eqs._diffeq_names_nonzero[0]))
        var_len  = len(dict.fromkeys(self.eqs._diffeq_names))+1 # +1 needed to store t
        
        for index,varname in enumerate(self.eqs._diffeq_names):
            self.index_to_varname.append(varname)
            self.varname_to_index[varname]= index
            if varname in self.eqs._diffeq_names_nonzero :
                self.index_nonzero.append(index)
        
        self.S_in = cuda.pagelocked_zeros((n,var_len),numpy.float32)
        
        self.S_out = cuda.pagelocked_zeros((n,var_len),numpy.float32)
        
        nbytes = n * var_len * numpy.dtype(numpy.float32).itemsize
        self.S_in_gpu = cuda.mem_alloc(nbytes)
        self.S_out_gpu = cuda.mem_alloc(nbytes)
        
        Z = zeros((n,var_len),numpy.float32)
        self.A_gpu = cuda.mem_alloc(nbytes)
        cuda.memcpy_htod(self.A_gpu, Z)
        self.B_gpu = cuda.mem_alloc(nbytes)
        cuda.memcpy_htod(self.B_gpu, Z)
        self.S_temp_gpu = cuda.mem_alloc(nbytes)
        
        modFun={}
        self.applyFun = {}
        for x in self.index_nonzero:
            s = self.eqs._function_C_String[self.index_to_varname[x]]
            args_fun =[]
            for i in xrange(var_len):
                args_fun.append("S_temp["+str(i)+" + blockIdx.x * var_len]")
            modFun[x] = SourceModule("""
                __device__ double f"""+ s +"""
                
                __global__ void applyFun(float *A,float *B,float *S_in,float *S_temp, int x, int var_len)
                { 
                    
                    int idx = x + blockIdx.x * var_len;
                    S_temp[idx] = 0;
                    B[idx] = f("""+",".join(args_fun)+""");
                    S_temp[idx] = 1;
                    A[idx] = f("""+",".join(args_fun)+""") - B[idx];
                    B[idx] /= A[idx];
                    S_temp[idx] = S_in[idx];
                }
                """)
            self.applyFun[x] = modFun[x].get_function("applyFun")
            self.applyFun[x].prepare(['P','P','P','P','i','i'],block=(1,1,1))
        
        self.calc_dict = {}
        self.already_calc = {}
    
    def __call__(self, P):
        
        if self.prepared==False:
            self.prepare(P)
            self.prepared = True
        '''
        Updates the state variables.
        Careful here: always use the slice operation for affectations.
        P is the neuron group.
        '''
        var_len  = len(dict.fromkeys(self.eqs._diffeq_names))+1 # +1 needed to store t
        
        if self.changed :
            if not self.firstTime:
                cuda.memcpy_dtoh(self.S_out, self.S_out_gpu)
                for var in self.eqs._diffeq_names_nonzero:
                    Temp = P.state_(var)
                    Temp[:] = self.S_out[:,self.varname_to_index[var]]
            else :
                self.firstTime = False
            
            for var in self.eqs._diffeq_names:
                self.S_in[:,self.varname_to_index[var]] = P.state_(var).astype(numpy.float32)
            if self._frozen:
                self.S_in[:,var_len-1]= numpy.float32(P.clock._t) # without units
            else:
                self.S_in[:,var_len-1] = numpy.float32(P.clock.t) #time
            cuda.memcpy_htod(self.S_in_gpu, self.S_in)
            cuda.memcpy_dtod(self.S_out_gpu,self.S_in_gpu, self.S_in.size * self.S_in.dtype.itemsize)
            cuda.memcpy_dtod(self.S_temp_gpu,self.S_in_gpu, self.S_in.size * self.S_in.dtype.itemsize)
        else :
            cuda.memcpy_dtod(self.S_in_gpu,self.S_out_gpu, self.S_in.size * self.S_in.dtype.itemsize)
            cuda.memcpy_dtod(self.S_temp_gpu,self.S_out_gpu, self.S_in.size * self.S_in.dtype.itemsize)
          
        self.gpu_exponential_euler(P.clock._dt)
        self.changed = False
    
    def updateV(self,v_gpu):
        n = len(self.S_in)
        var_len = len(self.S_in[0])
        var_lenBis = numpy.int32(var_len)
        vx = numpy.int32(self.varname_to_index['v'])
        self.getV.prepared_call((n,1),self.S_out_gpu,v_gpu,vx,var_lenBis)
    
    def gpu_exponential_euler(self,dt):
        n = len(self.S_in)
        var_len = len(self.S_in[0])
        var_lenBis = numpy.int32(var_len)
        dt = numpy.float32(dt)
         
        for x in self.index_nonzero:
            x = numpy.int32(x)
            self.applyFun[x].prepared_call((n,1),self.A_gpu, self.B_gpu, self.S_in_gpu, self.S_temp_gpu, x, var_lenBis)    
        
        self.func.prepared_call((n,1),self.A_gpu,self.B_gpu,self.S_in_gpu,self.S_out_gpu,dt)
        
    def calc(self,var,res_gpu):
        n = len(self.S_in)
        var_len = len(self.S_in[0])
        
        if var not in self.already_calc:
            vars_strings = []
            for var_aux in self.eqs._diffeq_names:
                vars_strings.append("float "+ var_aux)
            vars_strings.append("float t")
            
            expr = self.eqs._string[var]
            namespace = self.eqs._namespace[var] # name space of the function
            all_variables = self.eqs._eq_names + self.eqs._diffeq_names + self.eqs._alias.keys() + ['t']
            expr = optimiser.freeze(expr, all_variables, namespace)
            s = "(" + ",".join(vars_strings) + ") { float result = " + expr + "; return result; }"
            for var_aux in self.eqs._diffeq_names: #this is ugly. really ugly.
                s = s.replace(var_aux+'**2',var_aux+'*'+var_aux)
                s = s.replace(var_aux+'**3',var_aux+'*'+var_aux+'*'+var_aux)
                s = s.replace(var_aux+'**4',var_aux+'*'+var_aux+'*'+var_aux+'*'+var_aux)
                s = s.replace(var_aux+'**5',var_aux+'*'+var_aux+'*'+var_aux+'*'+var_aux+'*'+var_aux)
            args_fun =[]
            for i in xrange(var_len):
                args_fun.append("S_out["+str(i)+" + blockIdx.x * var_len]")
            mod = SourceModule("""
                    __device__ float f"""+ s +"""
                    
                    __global__ void calc(float *res,float *S_out, int var_len)
                    { 
                        int idx = blockIdx.x;
                        res[idx] = f("""+",".join(args_fun)+""");
                        
                    }
                    """)
            self.calc_dict[var] = mod.get_function("calc")
            self.calc_dict[var].prepare(['P','P','i'],block=(1,1,1))
            self.already_calc[var] = True
        self.calc_dict[var].prepared_call((n,1),res_gpu,self.S_out_gpu,numpy.int32(var_len))
    
    def compile_functions_C(self, eqs, freeze=False):
        """
        Compile all functions defined as strings.
        If freeze is True, all external parameters and units are replaced by their value.
        ALL FUNCTIONS MUST HAVE STRINGS.
        """
        all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
        # Check if freezable
        freeze = freeze and all([optimiser.freeze(expr, all_variables, eqs._namespace[name])\
                               for name, expr in eqs._string.iteritems()])
        eqs._frozen = freeze
        
        vars_strings = []
        for var in eqs._diffeq_names:
            vars_strings.append("float "+ var)
        vars_strings.append("float t")
        # Compile strings to functions
        for name, expr in eqs._string.iteritems():
            namespace = eqs._namespace[name] # name space of the function
            expr = optimiser.freeze(expr, all_variables, namespace)
            s = "(" + ",".join(vars_strings) + ") {float result = " + expr + "; return result; }"
            eqs._function_C_String[name] = s