from brian import *
import brian.optimiser as optimiser
from scipy import weave
import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import time

DEBUG_BUFFER_CACHE = False

class GPUNonlinearStateUpdater(NonlinearStateUpdater):
    def __init__(self,eqs,clock=None,freeze=False):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.clock_dt = float(guess_clock(clock).dt)
        self.code_gpu = self.generate_forward_euler_code()
        self._prepared = False
        
    def generate_forward_euler_code(self):
        eqs = self.eqs
        M = len(eqs._diffeq_names)
        all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
        clines = '__global__ void stateupdate(int N, double t, double *S)\n'
        clines += '{\n'
        clines += '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        clines += '    if(i>=N) return;\n'
        for j, name in enumerate(eqs._diffeq_names):
            clines += '    double &' + name + ' = S[i+'+str(j)+'*N];\n'
        for name in eqs._eq_names:
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            clines += '    double '+name+'__tmp = '+expr+';\n'
        for j, name in enumerate(eqs._diffeq_names):
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            if name in eqs._diffeq_names_nonzero:
                clines += '    double '+name+'__tmp = '+expr+';\n'
        for name in eqs._diffeq_names_nonzero:
            clines += '    '+name+' += '+str(self.clock_dt)+'*'+name+'__tmp;\n'
        clines += '}\n'
        self.gpu_mod = drv.SourceModule(clines)
        self.gpu_func = self.gpu_mod.get_function("stateupdate")
        return clines
    
    def __call__(self, P):
        if not self._prepared:
            self._args = [int32(len(P)), float64(0.0), P._S_gpu]
            blocksize = 512
            if len(P)<512:
                blocksize = len(P)
            if len(P)%blocksize==0:
                gridsize = len(P)/blocksize
            else:
                gridsize = len(P)/blocksize+1
            self._prepared = True
            self.gpu_func.prepare((int32, float64, 'i'), (blocksize,1,1))
            self._S_gpu_addr = int(P._S_gpu.gpudata)
            self._gpu_N = int32(len(P))
            self._gpu_grid = (gridsize,1)
        self.gpu_func.prepared_call(self._gpu_grid, self._gpu_N, float64(P.clock._t), self._S_gpu_addr)

class GPUNeuronGroup(NeuronGroup):
    def __init__(self, N, eqs, clock=None):
        eqs.prepare()
        NeuronGroup.__init__(self, N, eqs, clock=clock)
        self.clock = guess_clock(clock)
        self._state_updater = GPUNonlinearStateUpdater(eqs, clock=self.clock)
        self._S_gpu = gpuarray.to_gpu(self._S)
        self._data_changed = False
        self._gpu_data_changed = False
        self._gpuneurongroup_init_finished = True
    
    def _copy_from_gpu(self):
        if not hasattr(self, '_gpuneurongroup_init_finished'): return
        if self._gpu_data_changed:
            object.__setattr__(self, '_gpu_data_changed', False)
            self._S_gpu.get(self._S)
            if DEBUG_BUFFER_CACHE:
                print 'copying from gpu'
    
    def _write_to_gpu(self):
        if not hasattr(self, '_gpuneurongroup_init_finished'): return
        if self._data_changed:
            object.__setattr__(self, '_data_changed', False)
            if DEBUG_BUFFER_CACHE:
                print 'writing to gpu'
            self._S_gpu.set(self._S)
    
    def update(self):
        self._write_to_gpu()
        NeuronGroup.update(self)
        self._gpu_data_changed = True
       
    def state_(self, name):
        try:
            self._gpuneurongroup_init_finished
            self._copy_from_gpu()
            return NeuronGroup.state_(self, name)
        except AttributeError:
            return NeuronGroup.state_(self, name)
    state = state_
    
    def __getattr__(self, name):
        try:
            object.__getattribute__(self, '_gpuneurongroup_init_finished')
            return NeuronGroup.__getattr__(self, name)
        except AttributeError:
            return object.__getattribute__(self, name) 
    def __setattr__(self, name, val):
        try:
            self._gpuneurongroup_init_finished
            NeuronGroup.__setattr__(self, name, val)
            if name in self.var_index:
                object.__setattr__(self, '_data_changed', True)
        except AttributeError:
            object.__setattr__(self, name, val)

if __name__=='__main__':
    
    #duration = 10*second
    #N = 1000
    #domonitor = False
    
    duration = 100*ms
    N = 100000
    domonitor = False
    showfinal = False
    
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
    
    G = GPUNeuronGroup(N, eqs)
    
    print 'GPU loop code:'
    print G._state_updater.code_gpu
    
    G.V = 1
    
    if domonitor:
        M = StateMonitor(G, 'V', record=True)
    
    start = time.time()
    run(duration)
    print 'GPU code:', (time.time()-start)*second
    if domonitor: M_V = M[0]

    if domonitor:
        plot(M.times, M_V)
    if showfinal:
        figure()
        plot(G.V)
    if domonitor or showfinal:
        show()
