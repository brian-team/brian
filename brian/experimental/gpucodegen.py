from brian import *
import brian.optimiser as optimiser
from scipy import weave
import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import time

class GPUNonlinearStateUpdater(NonlinearStateUpdater):
    def __init__(self,eqs,clock=None,freeze=False):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.clock_dt = float(guess_clock(clock).dt)
        self.code_gpu = self.generate_forward_euler_code()
        self._prepared = False
        
    def generate_forward_euler_code(self):
        eqs = self.eqs
        all_variables = eqs._eq_names+eqs._diffeq_names+eqs._alias.keys()+['t']
        clines = '__global__ void stateupdate(double t, ' + ', '.join(['double *'+name for name in eqs._diffeq_names]) + ')\n'
        clines += '{\n'
        clines += '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        for name in eqs._eq_names:
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            for dename in eqs._diffeq_names:
                expr = expr.replace(dename, dename+'[i]')
            clines += '    double '+name+'__tmp = '+expr+';\n'
        for j, name in enumerate(eqs._diffeq_names):
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            for dename in eqs._diffeq_names:
                expr = expr.replace(dename, dename+'[i]')
            if name in eqs._diffeq_names_nonzero:
                clines += '    double '+name+'__tmp = '+expr+';\n'
        for name in eqs._diffeq_names_nonzero:
            clines += '    '+name+'[i] += '+str(self.clock_dt)+'*'+name+'__tmp;\n'
        clines += '}\n'
        self.gpu_mod = drv.SourceModule(clines)
        self.gpu_func = self.gpu_mod.get_function("stateupdate")
        return clines
    
    def __call__(self, P):
        if not self._prepared:
            self._args = [float64(0.0)]+[P._gpu_vars[name] for name in self.eqs._diffeq_names]
            self._stream = drv.Stream()
            blocksize = 512
            if len(P)<512:
                blocksize = len(P)
            if len(P)%blocksize==0:
                gridsize = len(P)/blocksize
            else:
                gridsize = len(P)/blocksize+1
            self._kwds = {'block':(blocksize,1,1),
                          'stream':self._stream,
                          'grid':(gridsize,1)}
            self._prepared = True
        #self._stream.synchronize()
        self._args[0] = float64(P.clock._t)
        self.gpu_func(*self._args, **self._kwds)

class GPUNeuronGroup(NeuronGroup):
    def __init__(self, N, eqs, clock=None):
        eqs.prepare()
        self.clock = guess_clock(clock)
        self._state_updater = GPUNonlinearStateUpdater(eqs, clock=self.clock)
        self._resetfun = NoReset()
        self._threshold = NoThreshold()
        self._spiking = False
        self._gpu_vars = dict((name, gpuarray.to_gpu(zeros(N))) for name in eqs._diffeq_names)
        self._owner = self
        self._N = N
        self.var_index = dict((name, -1) for name in eqs._eq_names+eqs._diffeq_names+eqs._alias.keys())
    
    def __len__(self):
        return self._N
    
    def state_(self, name):
        return self._gpu_vars[name].get()
    state = state_
    
    def unit(self, name):
        return 1
    
    def __setattr__(self, name, val):
        if hasattr(self, '_gpu_vars') and name in self._gpu_vars:
            if not isinstance(val, ndarray):
                val = val*ones(self._N)
            self._gpu_vars[name].set(val)
        else:
            NeuronGroup.__setattr__(self, name, val)

if __name__=='__main__':
    
    #duration = 10*second
    #N = 1000
    #domonitor = False
    
    duration = 100*ms
    N = 10
    domonitor = True
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
    print eqs
    
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
