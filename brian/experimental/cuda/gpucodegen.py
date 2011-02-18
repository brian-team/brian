#import brian_no_units
from brian import *
from brian import optimiser
from scipy import weave
import pycuda
import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
from pycuda import gpuarray
from buffering import *
import time
from brian.experimental.codegen.rewriting import rewrite_to_c_expression

__all__ = ['GPUNonlinearStateUpdater', 'UserControlledGPUNonlinearStateUpdater',
           'GPUNeuronGroup']

#DEBUG_BUFFER_CACHE = False

class GPUNonlinearStateUpdater(NonlinearStateUpdater):
    def __init__(self, eqs, clock=None, freeze=False, precision='double', maxblocksize=512, forcesync=False):
        NonlinearStateUpdater.__init__(self, eqs, clock, compile=False, freeze=freeze)
        self.precision = precision
        if self.precision == 'double':
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
        all_variables = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
        clines = '__global__ void stateupdate(int N, SCALAR t, SCALAR *S)\n'
        clines += '{\n'
        clines += '    int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
        clines += '    if(i>=N) return;\n'
        for j, name in enumerate(eqs._diffeq_names):
            clines += '    int _index_' + name + ' = i+' + str(j) + '*N;\n'
        for j, name in enumerate(eqs._diffeq_names):
#            clines += '    SCALAR &' + name + ' = S[i+'+str(j)+'*N];\n'
            clines += '    SCALAR ' + name + ' = S[_index_' + name + '];\n'
        for j, name in enumerate(eqs._diffeq_names):
            namespace = eqs._namespace[name]
            expr = optimiser.freeze(eqs._string[name], all_variables, namespace)
            expr = rewrite_to_c_expression(expr)
            print expr
            if name in eqs._diffeq_names_nonzero:
                clines += '    SCALAR ' + name + '__tmp = ' + expr + ';\n'
        for name in eqs._diffeq_names_nonzero:
#            clines += '    '+name+' += '+str(self.clock_dt)+'*'+name+'__tmp;\n'
            clines += '    S[_index_' + name + '] = ' + name + '+' + str(self.clock_dt) + '*' + name + '__tmp;\n'
        clines += '}\n'
        clines = clines.replace('SCALAR', self.precision)
        self.gpu_mod = compiler.SourceModule(clines)
        self.gpu_func = self.gpu_mod.get_function("stateupdate")
        return clines

    def _prepare(self, P):
        blocksize = self.maxblocksize
        if len(P) < blocksize:
            blocksize = len(P)
        if len(P) % blocksize == 0:
            gridsize = len(P) / blocksize
        else:
            gridsize = len(P) / blocksize + 1
        self._prepared = True
        self.gpu_func.prepare((int32, self.precision_dtype, 'i'), (blocksize, 1, 1))
        self._S_gpu_addr = P._S.gpu_pointer
        self._gpu_N = int32(len(P))
        self._gpu_grid = (gridsize, 1)

    def __call__(self, P):
        if not self._prepared:
            self._prepare(P)
        P._S.sync_to_gpu()
        self.gpu_func.prepared_call(self._gpu_grid, self._gpu_N, self.precision_dtype(P.clock._t), self._S_gpu_addr)
        P._S.changed_gpu_data()
        if self.forcesync:
            P._S.sync_to_cpu()
            P._S.changed_cpu_data()


class UserControlledGPUNonlinearStateUpdater(GPUNonlinearStateUpdater):
    def __init__(self, eqs, clock=None, freeze=False, precision='double', maxblocksize=512, gpu_to_cpu_vars=None, cpu_to_gpu_vars=None):
        GPUNonlinearStateUpdater.__init__(self, eqs, clock=clock, freeze=freeze, precision=precision, maxblocksize=maxblocksize, forcesync=False)
        self.gpu_to_cpu_vars = gpu_to_cpu_vars
        self.cpu_to_gpu_vars = cpu_to_gpu_vars

    def _prepare(self, P):
        GPUNonlinearStateUpdater._prepare(self, P)
        if isinstance(P._S._gpu_arr, pycuda.gpuarray.GPUArray):
            self._gpuoffset = int(P._S._gpu_arr.gpudata)
        elif isinstance(P._S._gpu_arr, pycuda.driver.DeviceAllocation):
            self._gpuoffset = int(P._S._gpu_arr)
        self._cpuflat = array(P._S, copy=False)
        self._cpuflat.shape = self._cpuflat.size

    def __call__(self, P):
        if not self._prepared:
            self._prepare(P)
        # copy from CPU to GPU
        for i, j, k in self.cpu_to_gpu_vars:
            pycuda.driver.memcpy_htod(self._gpuoffset + i, self._cpuflat[j:k])
        self.gpu_func.prepared_call(self._gpu_grid, self._gpu_N, self.precision_dtype(P.clock._t), self._S_gpu_addr)
        # copy from GPU back to CPU
        for i, j, k in self.gpu_to_cpu_vars:
            pycuda.driver.memcpy_dtoh(self._cpuflat[j:k], self._gpuoffset + i)
        P._S._cpu_data_changed = False # override any buffering
        P._S._gpu_data_changed = False # override any buffering


class GPUNeuronGroup(NeuronGroup):
    '''
    Neuron group which performs numerical integration on the GPU.
    
    .. warning::
        This class is still experimental, not supported and subject to change
        in future versions of Brian.
        
    Initialised with arguments as for :class:`NeuronGroup` and additionally:
    
    ``precision='double'``
        The GPU scalar precision to use, older models can only use
        ``precision='float'``.
    ``maxblocksize=512``
        If GPU compilation fails, reduce this value.
    ``forcesync=False``
        Whether or not to force copying of state variables to and from the
        GPU each time step. This is slow, so it is better to specify precisely
        which variables should be copied to and from using ``gpu_to_cpu_vars``
        and ``cpu_to_gpu_vars``.
    ``pagelocked_mem=True``
        Whether to store state variables in pagelocked memory on the CPU,
        which makes copying data to/from the GPU twice as fast.
    ``cpu_to_gpu_vars=None``, ``gpu_to_cpu_vars=None``
        Which variables should be copied each time step from the CPU to the GPU
        (before state update) and from the GPU to the CPU (after state update).
        
    The point of the copying of variables to and from the GPU is that the GPU
    maintains a separate memory from the CPU, and so changes made on either the
    CPU or GPU won't automatically be reflected in the other. Since only
    numerical integration is done on the GPU, any state variable that is
    modified by incoming synapses, for example, should be copied to and from
    the GPU each time step. In addition, any variables used for thresholding
    or resetting need to be appropriately copied (GPU->CPU for thresholding, and
    both for resetting).
    '''
    def __init__(self, N, model, threshold=None, reset=NoReset(),
                 init=None, refractory=0 * msecond, level=0,
                 clock=None, order=1, implicit=False, unit_checking=True,
                 max_delay=0 * msecond, compile=False, freeze=False, method=None,
                 precision='double', maxblocksize=512, forcesync=False, pagelocked_mem=True,
                 gpu_to_cpu_vars=None, cpu_to_gpu_vars=None):
        eqs = model
        eqs.prepare()
        NeuronGroup.__init__(self, N, eqs, threshold=threshold, reset=reset,
                             init=init, refractory=refractory, level=level,
                             clock=clock, order=order, compile=compile, freeze=freeze, method=method)
        self.precision = precision
        if self.precision == 'double':
            self.precision_dtype = float64
            self.precision_nbytes = 8
        else:
            self.precision_dtype = float32
            self.precision_nbytes = 4
        self.clock = guess_clock(clock)
        if gpu_to_cpu_vars is None and cpu_to_gpu_vars is None:
            self._state_updater = GPUNonlinearStateUpdater(eqs, clock=self.clock, precision=precision, maxblocksize=maxblocksize,
                                                           forcesync=forcesync)
        else:
            cpu_to_gpu_vars = [(self.get_var_index(var) * len(self) * self.precision_nbytes,
                                self.get_var_index(var) * len(self),
                                (self.get_var_index(var) + 1) * len(self)) for var in cpu_to_gpu_vars]
            gpu_to_cpu_vars = [(self.get_var_index(var) * len(self) * self.precision_nbytes,
                                self.get_var_index(var) * len(self),
                                (self.get_var_index(var) + 1) * len(self)) for var in gpu_to_cpu_vars]
            self._state_updater = UserControlledGPUNonlinearStateUpdater(eqs, clock=self.clock, precision=precision, maxblocksize=maxblocksize,
                                                           gpu_to_cpu_vars=gpu_to_cpu_vars, cpu_to_gpu_vars=cpu_to_gpu_vars)
        if pagelocked_mem:
            self._S = GPUBufferedArray(drv.pagelocked_zeros(self._S.shape, dtype=self.precision_dtype))
        else:
            self._S = GPUBufferedArray(array(self._S, dtype=self.precision_dtype))
        self._gpuneurongroup_init_finished = True

    def copyvar_cpu_to_gpu(self, var):
        i, j, k = (self.get_var_index(var) * len(self) * self.precision_nbytes,
                   self.get_var_index(var) * len(self),
                   (self.get_var_index(var) + 1) * len(self))
        pycuda.driver.memcpy_htod(self._state_updater._gpuoffset + i, self._state_updater._cpuflat[j:k])

    def copyvar_gpu_to_cpu(self, var):
        i, j, k = (self.get_var_index(var) * len(self) * self.precision_nbytes,
                   self.get_var_index(var) * len(self),
                   (self.get_var_index(var) + 1) * len(self))
        pycuda.driver.memcpy_dtoh(self._state_updater._cpuflat[j:k], self._state_updater._gpuoffset + i)

    def __setattr__(self, name, val):
        try:
            self._gpuneurongroup_init_finished
            NeuronGroup.__setattr__(self, name, val)
            if name in self.var_index:
                self._S.changed_cpu_data()
        except AttributeError:
            object.__setattr__(self, name, val)

if __name__ == '__main__':

    from brian.experimental.ccodegen import AutoCompiledNonlinearStateUpdater
    set_global_preferences(usecodegen=False)

    #duration = 10*second
    #N = 1000
    #domonitor = False

    duration = 1000 * ms
    N = 100
    domonitor = False
    showfinal = False
    forcesync = True
    method = 'gpu' # methods are 'c', 'python' and 'gpu'

    if drv.get_version() == (2, 0, 0): # cuda version
        precision = 'float'
    elif drv.get_version() > (2, 0, 0):
        precision = 'double'
    else:
        raise Exception, "CUDA 2.0 required"
    #precision = 'float'
    import buffering
    buffering.DEBUG_BUFFER_CACHE = False

#    eqs = Equations('''
#    #dV/dt = -V*V/(10*ms) : 1
#    dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
#    #dV/dt = -V*V*V*V*V/(100*ms) : 1
#    #dW/dt = -W*W*W*W*W/(100*ms) : 1
#    #dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
#    #dW/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
#    #dW2/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
#    #dV/dt = h/(10*ms) : 1
#    #h = -V*V : 1
#    ''')
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
    
    from brian.library.ionic_currents import *

    El = 10.6 * mV
    EK = -12 * mV
    ENa = 120 * mV
    eqs = MembraneEquation(1 * uF) + leak_current(.3 * msiemens, El)
    eqs += K_current_HH(36 * msiemens, EK) + Na_current_HH(120 * msiemens, ENa)
    eqs += Current('I:amp')
    #eqs.prepare()
    #
    #for n in eqs._string.keys():
    #    eqs._string[n] = rewrite_to_c_expression(eqs._string[n])
    #print eqs


    if method == 'gpu':
        G = GPUNeuronGroup(N, eqs, precision=precision, maxblocksize=256, forcesync=forcesync)

        print 'GPU loop code:'
        print G._state_updater.code_gpu
        gf = G._state_updater.gpu_func
        print '(lmem, smem, registers) = ', (gf.local_size_bytes, gf.shared_size_bytes, gf.num_regs)
        devdata = pycuda.tools.DeviceData()
        orec = pycuda.tools.OccupancyRecord(devdata, 256)
        print 'tb_per_mp', orec.tb_per_mp
        print 'limited_by', orec.limited_by
        print 'warps_per_mp', orec.warps_per_mp
        print 'occupancy', orec.occupancy
    elif method == 'c':
        G = NeuronGroup(N, eqs, compile=True, freeze=True)
        su = AutoCompiledNonlinearStateUpdater(eqs, G.clock, freeze=True)
        G._state_updater = su
    elif method == 'python':
        #G = NeuronGroup(N, eqs, freeze=True)#, compile=True, freeze=True)
        G = NeuronGroup(N, eqs, compile=True, freeze=True)

    G.V = 1

    if domonitor:
        M = StateMonitor(G, 'V', record=True)

    start = time.time()
    run(duration)
    autoinit.context.synchronize()
    print method, 'code:', (time.time() - start)
    if domonitor: M_V = M[0]

    if domonitor:
        plot(M.times, M_V)
    if showfinal:
        figure()
        plot(G.V)
    if domonitor or showfinal:
        show()
