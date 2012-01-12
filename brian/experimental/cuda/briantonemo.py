from brian import *
import nemo
import ctypes
try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    import pycuda.compiler as compiler
    from pycuda import gpuarray
except ImportError, e:
    print e
    pycuda = None
    log_warn('brian.experimental.cuda.gpucodegen', 'Cannot import pycuda')
from brian.experimental.codegen.rewriting import rewrite_to_c_expression

use_delay_connection = False

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

_created_network = False

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
        if pycuda is not None:
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
        if use_delay_connection:
            DelayConnection.do_propagate(self)
            return
        if not self.iscompressed:
            self.compress()
        if self.nemo_use_gpu:
            self.propagate(self.source.get_spikes(0))
        else:
            self.propagate(self.source.get_spikes(0))
        
    def propagate(self, spikes):
        if use_delay_connection:
            DelayConnection.propagate(self, spikes)
            return
        if self.nemo_use_gpu:
            spikes_bool = zeros(len(self.source), dtype=uint32)
            spikes_bool[spikes] = True
            spikes_gpu = pycuda.gpuarray.to_gpu(spikes_bool)
            spikes_gpu_ptr = int(int(spikes_gpu.gpudata))
            exc_ptr, inh_ptr = tuple(self.nemo_sim.propagate(spikes_gpu_ptr, len(self.source)))
            exc = zeros(len(self.source), dtype=float32)
            inh = zeros(len(self.source), dtype=float32)
            pycuda.driver.memcpy_dtoh(exc, exc_ptr)
            pycuda.driver.memcpy_dtoh(inh, inh_ptr)
            self.target._S[self.nstate] += exc
            self.target._S[self.nstate] += inh
        else:
            spikes = array(spikes, dtype=uint32)
            spikes_ptr = spikes.ctypes.data
            spikes_len = len(spikes)
            exc_ptr, inh_ptr = tuple(self.nemo_sim.propagate(spikes_ptr, spikes_len))
            exc = numpy_array_from_memory(exc_ptr, len(self.source), float32)
            inh = numpy_array_from_memory(inh_ptr, len(self.source), float32)
            self.target._S[self.nstate] += exc
            self.target._S[self.nstate] += inh


class NemoNetworkPropagate(NetworkOperation):
    def __init__(self, net):
        self.net = net
        self.when = 'after_connections'
    def __call__(self):
        spikes = hstack(self.net.nemo_spikes, dtype=uint32)
        self.net.nemo_spikes = []
        spikes_ptr = spikes.ctypes.data
        spikes_len = len(spikes)
        exc_ptr, inh_ptr = tuple(self.nemo_sim.propagate(spikes_ptr, spikes_len))
        N = self.net.total_neurons
        exc = numpy_array_from_memory(exc_ptr, N, float32)
        inh = numpy_array_from_memory(inh_ptr, N, float32)
        for G in self.net.groups:
            target_offset = self.net.group_offset[id(G)]
            #TODO: need to handle different target states with different neurons
            #self.target._S[self.nstate] += exc
            #self.target._S[self.nstate] += inh

class NemoNetworkConnectionPropagate(object):
    def __init__(self, net, source_offset, target_offset):
        set.net = net
        self.source_offset = source_offset
        self.target_offset = target_offset
    def __call__(self, C, spikes):
        self.net.nemo_spikes.append(spikes+self.source_offset)

class NemoNetwork(Network):
    def prepare(self):
        global _created_network
        if _created_network:
            raise NotImplementedError("Current version only supports a single run.")
        _created_network = True

        # add a NetworkOperation that will be used to carry out the propagation
        # by NeMo
        nemo_propagate = NemoNetworkPropagate(self)
        self.add(nemo_propagate)
        self.nemo_spikes = []
        
        Network.prepare(self)
        
        if hasattr(self, 'clocks') and len(self.clocks)>1:
            raise NotImplementedError("Current version only supports a single clock.")
        
        # combine all groups into one meta-group for NeMo, store the offsets
        self.group_offset = group_offset = {}
        self.total_neurons = total_neurons = 0
        for G in self.groups:
            group_offset[id(G)] = total_neurons
            total_neurons += len(G)

        # now upload to nemo
        self.nemo_net = nemo.Network()
        if pycuda is not None:
            self.nemo_use_gpu = False
            log_warn('brian.experimental.cuda.briantonemo',
                     'GPU available but not yet supported, using CPU.')
        else:
            self.nemo_use_gpu = False

        # create dummy neurons
        self.nemo_input_neuron_idx = self.nemo_net.add_neuron_type('Input')
        self.nemo_net.add_neuron(self.nemo_input_neuron_idx,
                                 range(total_neurons))

        # add connections and upload synapses to nemo
        for C in self.connections:
            # check limitations
            if C.__class__ is not DelayConnection:
                raise NotImplementedError("Only DelayConnections supported at the moment.")
            if not isinstance(C.W[0, :], SparseConnectionVector):
                raise NotImplementedError("Current version only supports sparse matrix types.")
            
            source_offset = group_offset[id(C.source)]
            target_offset = group_offset[id(C.target)]
            dt = C.source.clock.dt
            C.propagate = NemoNetworkConnectionPropagate(self, source_offset,
                                                         target_offset)
            # create synapses
            for i in xrange(len(C.source)):
                Wrow = self.W[i, :]
                Wdelay = self.delay[i, :]
                ind = (Wrow.ind+target_offset).tolist()
                delay = asarray(Wdelay/dt, dtype=int).tolist()
                if amax(delay)>=64:
                    raise NotImplementedError("Current version of NeMo has a maximum delay of 64 steps.")
                weight = asarray(Wrow, dtype=float32).tolist()
                if len(ind):
                    self.nemo_net.add_synapse(i+source_offset, ind, delay,
                                              weight, False)

        # configure
        self.nemo_conf = nemo.Configuration()
        if self.nemo_use_gpu:
            self.nemo_conf.set_cuda_backend(0)
        else:
            self.nemo_conf.set_cpu_backend()
        # simulation object
        self.nemo_sim = nemo.Simulation(self.nemo_net, self.nemo_conf)
