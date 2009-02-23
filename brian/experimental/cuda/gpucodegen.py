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
    def __init__(self, eqs, clock=None, freeze=False, precision='double', maxblocksize=512, forcesync=False):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.precision = precision
        if self.precision=='double':
            self.precision_dtype = float64
        else:
            self.precision_dtype = float32
        self.clock_dt = float(guess_clock(clock).dt)
        self.code_gpu = self.generate_forward_euler_code()
        self.maxblocksize = maxblocksize
        self.forcesync = forcesync
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
            blocksize = self.maxblocksize
            if len(P)<blocksize:
                blocksize = len(P)
            if len(P)%blocksize==0:
                gridsize = len(P)/blocksize
            else:
                gridsize = len(P)/blocksize+1
            self._prepared = True
            self.gpu_func.prepare((int32, self.precision_dtype, 'i'), (blocksize,1,1))
            self._S_gpu_addr = P._S.gpu_pointer
            self._gpu_N = int32(len(P))
            self._gpu_grid = (gridsize,1)
        P._S.sync_to_gpu()
        self.gpu_func.prepared_call(self._gpu_grid, self._gpu_N, self.precision_dtype(P.clock._t), self._S_gpu_addr)
        P._S.changed_gpu_data()
        if self.forcesync:
            P._S.sync_to_cpu()
            P._S.changed_cpu_data()

class GPUNeuronGroup(NeuronGroup):
    def __init__(self, N, model, threshold=None, reset=NoReset(),
                 init=None, refractory=0*msecond, level=0,
                 clock=None, order=1, implicit=False, unit_checking=True,
                 max_delay=0*msecond, compile=False, freeze=False, method=None,
                 precision='double', maxblocksize=512, forcesync=False, pagelocked_mem=True):
        eqs = model
        eqs.prepare()
        NeuronGroup.__init__(self, N, eqs, threshold=threshold, reset=reset,
                             init=init, refractory=refractory, level=level,
                             clock=clock, order=order, compile=compile, freeze=freeze, method=method)
        self.precision = precision
        if self.precision=='double':
            self.precision_dtype = float64
        else:
            self.precision_dtype = float32
        self.clock = guess_clock(clock)
        self._state_updater = GPUNonlinearStateUpdater(eqs, clock=self.clock, precision=precision, maxblocksize=maxblocksize,
                                                       forcesync=forcesync)
        if pagelocked_mem:
            self._S = GPUBufferedArray(drv.pagelocked_zeros(self._S.shape, dtype=self.precision_dtype))
        else:
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
    N = 1000
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
    #eqs = Equations('\n'.join('dv'+str(i)+'/dt=-v'+str(i)+'/second:1' for i in range(20))) #10 works 11 is too much
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
    gf = G._state_updater.gpu_func
    print '(lmem, smem, registers) = ', (gf.lmem, gf.smem, gf.registers)
    
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