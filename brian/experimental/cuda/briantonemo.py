from brian import *
from brian.utils.dynamicarray import *
import nemo
import ctypes
try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    import pycuda.compiler as compiler
    from pycuda import gpuarray
except ImportError, e:
    pycuda = None
from brian.experimental.codegen.rewriting import rewrite_to_c_expression

use_delay_connection = False

__all__ = ['NemoConnection', 'NemoNetwork']

def numpy_array_from_memory(ptr, N, dtype):
    '''
    Creates a numpy array of length N and given dtype from the pointer ptr.
    '''
    buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
    buffer_from_memory.restype = ctypes.py_object
    buffer = buffer_from_memory(ptr, dtype().itemsize*N)
    return frombuffer(buffer, dtype=dtype)

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
            delay = asarray(Wdelay/ms, dtype=int)+1
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
    def __init__(self, net, clock):
        NetworkOperation.__init__(self, when='after_connections', clock=clock)
        self.net = net
        self.collected_spikes = DynamicArray(0, dtype=uint32)
        self.nspikes = 0
    def prepare(self):
        self.connection_mappings = []
        for C in self.net.connections:
            if isinstance(C, SpikeMonitor):
                continue
            target_offset = self.net.group_offset[(id(C.target._owner), C.nstate)]
            targetvar = C.target._S[C.nstate]
            targetslice = slice(target_offset, target_offset+len(C.target))
            self.connection_mappings.append((targetvar, targetslice))
#            C.target._S[C.nstate] += exc[target_offset:target_offset+len(C.target)]
#            C.target._S[C.nstate] += inh[target_offset:target_offset+len(C.target)]
            
    def addspikes(self, spikes):
        if self.nspikes+len(spikes)>len(self.collected_spikes):
            self.collected_spikes.resize(self.nspikes+len(spikes))
        self.collected_spikes[self.nspikes:self.nspikes+len(spikes)] = spikes
        self.nspikes += len(spikes)
    def __call__(self):
        spikes = self.collected_spikes[:self.nspikes]
        spikes_ptr = spikes.ctypes.data
        spikes_len = len(spikes)
        def do_the_nemo_bit():
            return tuple(self.net.nemo_sim.propagate(spikes_ptr, spikes_len))
        exc_ptr, inh_ptr = do_the_nemo_bit()
        #exc_ptr, inh_ptr = tuple(self.net.nemo_sim.propagate(spikes_ptr, spikes_len))
        def do_dans_bit():
            N = self.net.total_neurons
            exc = numpy_array_from_memory(exc_ptr, N, float32)
            inh = numpy_array_from_memory(inh_ptr, N, float32)
            for targetvar, targetslice in self.connection_mappings:
                targetvar += exc[targetslice]
                targetvar += inh[targetslice]
#            for C in self.net.connections:
#                if isinstance(C, SpikeMonitor):
#                    continue
#                target_offset = self.net.group_offset[(id(C.target._owner), C.nstate)]
#                C.target._S[C.nstate] += exc[target_offset:target_offset+len(C.target)]
#                C.target._S[C.nstate] += inh[target_offset:target_offset+len(C.target)]
            self.nspikes = 0
        do_dans_bit()

class NemoNetworkConnectionPropagate(object):
    def __init__(self, net, source_offset):
        self.net = net
        self.source_offset = source_offset
    def __call__(self, spikes):
        self.net.nemo_propagate.addspikes(spikes+self.source_offset)


class NemoNetwork(Network):
    def prepare(self):
        global _created_network
        if _created_network:
            raise NotImplementedError("Current version only supports a single run.")
        _created_network = True

        for k, v in self._operations_dict.iteritems():
            v = [f for f in v if not (hasattr(f, '__name__') and f.__name__=='delayed_propagate')]
            self._operations_dict[k] = v
        Network.prepare(self)
        if hasattr(self, 'clocks') and len(self.clocks)>1:
            raise NotImplementedError("Current version only supports a single clock.")

        # add a NetworkOperation that will be used to carry out the propagation
        # by NeMo
        nemo_propagate = NemoNetworkPropagate(self, self.clock)
        self.nemo_propagate = nemo_propagate
        self.add(nemo_propagate)
                
        # combine all groups into one meta-group for NeMo, store the offsets
        self.group_offset = group_offset = {}
        total_neurons = 0
        for G in self.groups:
            group_offset[id(G)] = total_neurons
            total_neurons += len(G)
        for C in self.connections:
            # check limitations
            if isinstance(C, SpikeMonitor):
                continue
            if C.__class__ is not DelayConnection:
                raise NotImplementedError("Only DelayConnections supported at the moment.")
            if not isinstance(C.W[0, :], SparseConnectionVector):
                raise NotImplementedError("Current version only supports sparse matrix types.")
            key = (id(C.target._owner), C.nstate)
            if key not in group_offset:
                group_offset[key] = total_neurons
                total_neurons += len(C.target)
        self.total_neurons = total_neurons

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
            if isinstance(C, SpikeMonitor):
                continue
            source_offset = group_offset[id(C.source._owner)]+C.source._origin
            target_offset = group_offset[(id(C.target._owner), C.nstate)]+C.target._origin
            dt = C.source.clock.dt
            C.propagate = NemoNetworkConnectionPropagate(self, source_offset)
            # create synapses
            for i in xrange(len(C.source)):
                Wrow = C.W[i, :]
                Wdelay = C.delay[i, :]
                ind = (Wrow.ind+target_offset).tolist()
                delay = (1+asarray(Wdelay/dt, dtype=int)).tolist()
                if amax(delay)>64:
                    raise NotImplementedError("Current version of NeMo has a maximum delay of 64 steps.")
                weight = asarray(Wrow, dtype=float32).tolist()
                if len(ind):
                    self.nemo_net.add_synapse(i+source_offset, ind, delay,
                                              weight, False)
        self._build_update_schedule()
        nemo_propagate.prepare()

        # configure
        self.nemo_conf = nemo.Configuration()
        if self.nemo_use_gpu:
            self.nemo_conf.set_cuda_backend(0)
        else:
            self.nemo_conf.set_cpu_backend()
        # simulation object
        self.nemo_sim = nemo.Simulation(self.nemo_net, self.nemo_conf)
