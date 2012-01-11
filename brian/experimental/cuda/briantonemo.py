from brian import *
import nemo
import ctypes
try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    import pycuda.compiler as compiler
    from pycuda import gpuarray
except ImportError:
    log_warn('brian.experimental.cuda.gpucodegen', 'Cannot import pycuda')
from brian.experimental.codegen.rewriting import rewrite_to_c_expression

__all__ = ['NemoConnection']

def numpy_array_from_memory(ptr, N, dtype):
    '''
    Creates a numpy array of length N and given dtype from the pointer ptr.
    '''
    buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
    buffer_from_memory.restype = ctypes.py_object
    buffer = buffer_from_memory(ptr, dtype().itemsize*N)
    return frombuffer(buffer, dtype)

_created_connection = False

class NemoConnection(DelayConnection):
    def compress(self):
        global _created_connection
        DelayConnection.compress(self)
        
        # check that limitations are met
        if _created_connection:
            raise NotImplementedError("Current version only supports a single connection.")
        _created_connection = True
        if self.source is not self.target:
            raise NotImplementedError("Current version only supports a single group.")
        if not isinstance(self.W[0, :], SparseConnectionVector):
            raise NotImplementedError("Current version only supports sparse matrix types.")

        # now upload to nemo
        self.nemo_net = nemo.Network()
        if False:#isinstance(self.source, GPUNeuronGroup):
            self.nemo_use_gpu = True
        else:
            self.nemo_use_gpu = False
        # create dummy neurons
        self.nemo_input_neuron_idx = self.nemo_net.add_neuron_type('Input')
        self.nemo_net.add_neuron(self.nemo_input_neuron_idx,
                                 range(len(self.source)))
        # create synapses
        for i in xrange(len(self.source)):
            Wrow = self.W[i, :]
            Wdelay = self.delay[i, :]
            ind = asarray(Wrow.ind)
            delay = asarray(Wdelay/ms, dtype=int)
            weight = asarray(Wrow, dtype=float32)
            if len(ind):
                self.nemo_net.add_synapse(i, ind.tolist(), delay.tolist(),
                                          weight.tolist(), False)
        # configure
        self.nemo_conf = nemo.Configuration()
        if self.nemo_use_gpu:
            self.nemo_conf.set_cuda_backend(0)
        else:
            self.nemo_conf.set_cpu_backend()
        # simulation object
        self.nemo_sim = nemo.Simulation(self.nemo_net, self.nemo_conf)

    def do_propagate(self):
        if not self.iscompressed:
            self.compress()
        if self.nemo_use_gpu:
            raise NotImplementedError("GPU version not yet implemented.")
        else:
            self.propagate(self.source.get_spikes(0))
        
    def propagate(self, spikes):
        spikes = array(spikes, dtype=uint32)
        if self.nemo_use_gpu:
            raise NotImplementedError("GPU version not yet implemented.")
        else:
            spikes_ptr = spikes.ctypes.data
            spikes_len = len(spikes)
            exc_ptr, inh_ptr = tuple(self.nemo_sim.propagate(spikes_ptr, spikes_len))
            exc = numpy_array_from_memory(exc_ptr, len(self.source), float32)
            inh = numpy_array_from_memory(inh_ptr, len(self.source), float32)
            self.target._S[self.nstate] += exc
            self.target._S[self.nstate] += inh
        #DelayConnection.propagate(self, spikes)

