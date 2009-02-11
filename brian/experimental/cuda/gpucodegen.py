from brian import *
import brian.optimiser as optimiser
from scipy import weave
import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from buffering import *
import time

__all__ = ['GPUNonlinearStateUpdater', 'GPUNeuronGroup']

#DEBUG_BUFFER_CACHE = False

class GPUNonlinearStateUpdater(NonlinearStateUpdater):
    def __init__(self, eqs, clock=None, freeze=False, precision='double'):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.precision = precision
        if self.precision=='double':
            self.precision_dtype = float64
        else:
            self.precision_dtype = float32
        self.clock_dt = float(guess_clock(clock).dt)
        self.code_gpu = self.generate_forward_euler_code()
        self._prepared = False
        
    def generate_forward_euler_code(self):
        eqs = self.eqs
        M = len(eqs._diffeq_names)
        all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
        clines = '__global__ void stateupdate(int N, SCALAR t, SCALAR *S)\n'
        clines += '{\n'
        clines += '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        clines += '    if(i>=N) return;\n'
        for j, name in enumerate(eqs._diffeq_names):
            clines += '    SCALAR &' + name + ' = S[i+'+str(j)+'*N];\n'
        for j, name in enumerate(eqs._diffeq_names):
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            if name in eqs._diffeq_names_nonzero:
                clines += '    SCALAR '+name+'__tmp = '+expr+';\n'
        for name in eqs._diffeq_names_nonzero:
            clines += '    '+name+' += '+str(self.clock_dt)+'*'+name+'__tmp;\n'
        clines += '}\n'
        clines = clines.replace('SCALAR', self.precision)
        self.gpu_mod = drv.SourceModule(clines)
        self.gpu_func = self.gpu_mod.get_function("stateupdate")
        return clines
    
    def __call__(self, P):
        if not self._prepared:
            #self._args = [int32(len(P)), float64(0.0), P._S.gpu_array]
            blocksize = 512
            if len(P)<512:
                blocksize = len(P)
            if len(P)%blocksize==0:
                gridsize = len(P)/blocksize
            else:
                gridsize = len(P)/blocksize+1
            self._prepared = True
            self.gpu_func.prepare((int32, self.precision_dtype, 'i'), (blocksize,1,1))
            self._S_gpu_addr = P._S.gpu_pointer#int(P._S_gpu.gpudata)
            self._gpu_N = int32(len(P))
            self._gpu_grid = (gridsize,1)
        P._S.sync_to_gpu()
        #print (self._gpu_grid, self._gpu_N, self.precision_dtype(P.clock._t), self._S_gpu_addr)
        self.gpu_func.prepared_call(self._gpu_grid, self._gpu_N, self.precision_dtype(P.clock._t), self._S_gpu_addr)
        P._S.changed_gpu_data()

class GPUNeuronGroup(NeuronGroup):
    def __init__(self, N, model, clock=None, precision='double'):
        eqs=model
        eqs.prepare()
        NeuronGroup.__init__(self, N, eqs, clock=clock)
        self.precision = precision
        if self.precision=='double':
            self.precision_dtype = float64
        else:
            self.precision_dtype = float32
        self.clock = guess_clock(clock)
        self._state_updater = GPUNonlinearStateUpdater(eqs, clock=self.clock, precision=precision)
        self._S = GPUBufferedArray(array(self._S, dtype=self.precision_dtype))
        self._gpuneurongroup_init_finished = True

    def __setattr__(self, name, val):
        try:
            self._gpuneurongroup_init_finished
            NeuronGroup.__setattr__(self, name, val)
            if name in self.var_index:
                self._S.changed_cpu_data()
        except AttributeError:
            object.__setattr__(self, name, val)

if __name__=='__main__':
    
    #duration = 10*second
    #N = 1000
    #domonitor = False
    
    duration = 100*ms
    N = 1000000
    domonitor = False
    showfinal = False
    if drv.get_version()==(2,0,0): # cuda version
        precision = 'float'
    elif drv.get_version()>(2,0,0):
        precision = 'double'
    else:
        raise Exception,"CUDA 2.0 required"
    #precision = 'float'
    import buffering
    buffering.DEBUG_BUFFER_CACHE = True
    
    eqs = Equations('''
    #dV/dt = -V*V/(10*ms) : 1
    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    dV/dt = W*W/(100*ms) : 1
    dW/dt = -V/(100*ms) : 1
    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dW/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dW2/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    #dV/dt = h/(10*ms) : 1
    #h = -V*V : 1
    ''')
    #print eqs

#    taum=20*ms
#    taue=5*ms
#    taui=10*ms
#    Vt=-50*mV
#    Vr=-60*mV
#    El=-49*mV
#    
#    eqs= Equations('''
#    dV/dt  = (ge+gi-(V-El))/taum : volt
#    dge/dt = -ge/taue : volt
#    dgi/dt = -gi/taui : volt
#    ''')
    
    G = GPUNeuronGroup(N, eqs, precision=precision)
    
    print 'GPU loop code:'
    print G._state_updater.code_gpu
    
    G.V = 1
    
    if domonitor:
        M = StateMonitor(G, 'V', record=True)
    
    start = time.time()
    run(duration)
    autoinit.context.synchronize()
    print 'GPU code:', (time.time()-start)*second
    if domonitor: M_V = M[0]

    if domonitor:
        plot(M.times, M_V)
    if showfinal:
        figure()
        plot(G.V)
    if domonitor or showfinal:
        show()