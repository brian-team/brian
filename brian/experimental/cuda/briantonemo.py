from brian import *
from brian.utils.dynamicarray import *
import new
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
                #self.nemo_net.add_synapse(i, ind, delay, weight, False)
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
        self.use_gpu = net.nemo_use_gpu
        
    def addspikes(self, spikes):
        if self.nspikes+len(spikes)>len(self.collected_spikes):
            self.collected_spikes.resize(self.nspikes+len(spikes))
        self.collected_spikes[self.nspikes:self.nspikes+len(spikes)] = spikes
        self.nspikes += len(spikes)
        
    def __call__(self):
        spikes = self.collected_spikes[:self.nspikes]
        total_neurons = self.net.total_neurons
        if self.use_gpu:
            if not hasattr(self, 'spikes_gpu'):
                spikes_bool = drv.pagelocked_zeros(total_neurons, dtype=uint32)
                spikes_bool[spikes] = True
                spikes_gpu = pycuda.gpuarray.to_gpu(spikes_bool)
                spikes_gpu_ptr = int(int(spikes_gpu.gpudata))
                self.spikes_bool = spikes_bool
                self.spikes_gpu = spikes_gpu
                self.spikes_gpu_ptr = spikes_gpu_ptr
            else:
                spikes_bool = self.spikes_bool
                spikes_bool[:] = False
                spikes_bool[spikes] = True
                spikes_gpu = self.spikes_gpu
                pycuda.driver.memcpy_htod(spikes_gpu.gpudata, spikes_bool)
                spikes_gpu_ptr = self.spikes_gpu_ptr
            acc_ptr = self.net.nemo_sim.propagate(self.synapse_type,
                                                  spikes_gpu_ptr, total_neurons)
            if not hasattr(self, 'acc'):
                self.acc = acc = drv.pagelocked_zeros(total_neurons, dtype=float32)
            else:
                acc = self.acc
            pycuda.driver.memcpy_dtoh(acc, acc_ptr)
        else:
            spikes_ptr = spikes.ctypes.data
            spikes_len = len(spikes)
            acc_ptr = self.net.nemo_sim.propagate(self.synapse_type,
                                                  spikes_ptr, spikes_len)
            acc = numpy_array_from_memory(acc_ptr, total_neurons, float32)
        for _, targetvar, targetslice in self.net.nemo_propagate_targets:
            targetvar += acc[targetslice]
        self.nspikes = 0

class NemoNetworkConnectionPropagate(object):
    def __init__(self, net, source_offset):
        self.net = net
        self.source_offset = source_offset
        
    def __call__(self, spikes):
        self.net.nemo_propagate.addspikes(spikes+self.source_offset)


def get_connection_variable(C):
    varname = str(C.nstate)
    for k, v in C.target.var_index.iteritems():
        if v==C.nstate and isinstance(k, str):
            varname = k
    return varname
    

class NemoNetwork(Network):
    def prepare(self):
        global _created_network
        if _created_network:
            raise NotImplementedError("Current version only supports a single network object.")
        _created_network = True

        Network.prepare(self)
        if hasattr(self, 'clocks') and len(self.clocks)>1:
            raise NotImplementedError("Current version only supports a single clock.")

        if pycuda is not None:
            self.nemo_use_gpu = True
#            self.nemo_use_gpu = False
#            log_warn('brian.experimental.cuda.briantonemo',
#                     'GPU available but not yet supported, using CPU.')
        else:
            self.nemo_use_gpu = False

        # add a NetworkOperation that will be used to carry out the propagation
        # by NeMo. The individual Connection objects push their spikes into a
        # global queue (this is done by NemoNetworkConnectionPropagate) and then
        # the NemoNetworkPropagate NetworkOperation object propagates the
        # accumulated effect of the spikes to the target NeuronGroup objects.
        nemo_propagate = NemoNetworkPropagate(self, self.clock)
        self.nemo_propagate = nemo_propagate
        # we add this object to the network and don't need to prepare() again
        # because we rebuild the update schedule below.
        self.add(nemo_propagate)
                
        # create virtual neurons for Nemo, the virtual neurons are either source
        # neurons or target neurons. For each Connection, we create a group of
        # source neurons (or re-use if it has already been created), and a
        # group of target neurons. Each target group and target variable has
        # a corresponding virtual group. In all cases, we work with the _owner
        # of the NeuronGroup so that we don't create multiple groups for each
        # subgroup.
        print 'Nemo network virtual neuron groups:'
        self.group_offset = group_offset = {}
        total_neurons = 0
        self.nemo_managed_connections = []
        self.nemo_propagate_targets = []
        network_ops_to_remove = []
        all_connections = []
        for C in self.connections:
            if isinstance(C, MultiConnection):
                all_connections.extend(C.connections)
            else:
                all_connections.append(C)
        for C in all_connections:
            # We handle only Connection and DelayConnection objects. For the CPU
            # version at least, we can rely on Brian to fall back to its
            # default behaviour otherwise.
            if C.__class__ is not DelayConnection and C.__class__ is not Connection:
                continue
            # Nemo cannot handle modulation, so we check for this case
            if C._nstate_mod is not None:
                log_warn("brian.experimental.cuda.briantonemo",
                         "Synaptic modulation is not supported in Nemo, skipped this Connection.")
                continue
            # Add the source neuron group
            G = C.source._owner
            if id(G) not in group_offset:
                group_offset[id(G)] = total_neurons
                print '- Source group, length', len(G), 'indices %d:%d'%(total_neurons, total_neurons+len(G))
                total_neurons += len(G)
            # Add the (target neuron group, target variable) virtual group              
            key = (id(C.target._owner), C.nstate)
            if key not in group_offset:
                group_offset[key] = total_neurons
                varname = get_connection_variable(C)
                print '- Target group, length', len(G), 'variable', varname, 'indices %d:%d'%(total_neurons, total_neurons+len(G))
                target_offset = total_neurons
                targetslice = slice(target_offset, target_offset+len(C.target))
                targetvar = C.target._S[C.nstate]
                self.nemo_propagate_targets.append((varname, targetvar, targetslice))
                total_neurons += len(C.target)
            self.nemo_managed_connections.append(C)
            if C.__class__ is DelayConnection:
                network_ops_to_remove.append(C.delayed_propagate)
        self.total_neurons = total_neurons

        # now upload to nemo
        self.nemo_net = nemo.Network()

        # create dummy synapse type
        self.synapse_type = self.nemo_net.add_synapse_type()
        self.nemo_propagate.synapse_type = self.synapse_type

        # create dummy neurons
        self.nemo_input_neuron_idx = self.nemo_net.add_neuron_type('Input',
                                                                   [self.synapse_type])
        self.nemo_net.add_neuron(self.nemo_input_neuron_idx,
                                 range(total_neurons))

        # add connections and upload synapses to nemo
        total_synapses = 0
        print 'Nemo network synapses:'
        for C in self.nemo_managed_connections:
            # We handle subgrouping using the source and target offset
            source_offset = group_offset[id(C.source._owner)]+C.source._origin
            target_offset = group_offset[(id(C.target._owner), C.nstate)]+C.target._origin
            dt = C.source.clock.dt
            # We replace the propagate method of the Connection with a nemo
            # version, which adds the spikes to a dynamic stack in the
            # NemoNetworkPropagate object, which is called after the connections
            # have been processed by Brian. Note that this way of doing things
            # only works for the CPU, for the GPU we will need to replace
            # do_propagate instead.
            C.propagate = NemoNetworkConnectionPropagate(self, source_offset)
            # create synapses
            this_connection_synapses = 0
            for i in xrange(len(C.source)):
                # Need to handle Connection/DelayConnection, and sparse/dense
                # matrix types
                Wrow = C.W[i, :]
                if C.__class__ is DelayConnection:
                    Wdelay = C.delay[i, :]
                else:
                    Wdelay = zeros(len(Wrow))
                if isinstance(Wrow, SparseConnectionVector):
                    ind = (Wrow.ind+target_offset)
                else:
                    ind = range(target_offset, target_offset+len(Wrow))
                delay = (1+asarray(Wdelay/dt, dtype=int))
                # Need to update this when NeMo gets support for longer delays
                if self.nemo_use_gpu and amax(delay)>512:
                    raise NotImplementedError("Current version of NeMo with GPU backend has a maximum delay of 512 steps.")
                if not self.nemo_use_gpu and amax(delay)>64:
                    raise NotImplementedError("Current version of NeMo with CPU backend has a maximum delay of 64 steps.")
                weight = asarray(Wrow, dtype=float32)
                total_synapses += len(weight)
                this_connection_synapses += len(weight)
                if len(ind):
                    self.nemo_net.add_synapse(self.synapse_type,
                                              i+source_offset, ind, delay,
                                              weight, False)
            print '-', C.__class__.__name__,
            print 'from source group of', len(C.source), 'neurons',
            print 'to target group of', len(C.target), 'neurons,',
            print 'variable %s,'%get_connection_variable(C),
            print 'matrix type %s,'%C.W.__class__.__name__,
            print 'number of synapses', this_connection_synapses

        # debug print the propagation targets
        print 'Nemo network propagation targets:'
        for varname, targetvar, targetslice in self.nemo_propagate_targets:
            print '- Variable', varname, 'indices %d:%d'%(targetslice.start, targetslice.stop)

        # remove the delayed_propagate functions which are used by
        # DelayConnection and will already have been inserted into the network
        # at this point (as they are in the contained_objects of
        # DelayConnection).
        for k, v in self._operations_dict.iteritems():
            v = [f for f in v if not f in network_ops_to_remove]
            self._operations_dict[k] = v
        
        # We changed lots of propagate functions so we need to rebuild the
        # update schedule to make use of them
        self._build_update_schedule()
        
        print 'Nemo network total neurons:', total_neurons
        print 'Nemo network total synapses:', total_synapses

        # configure
        self.nemo_conf = nemo.Configuration()
        if self.nemo_use_gpu:
            self.nemo_conf.set_cuda_backend(0)
        else:
            self.nemo_conf.set_cpu_backend()
        # simulation object
        self.nemo_sim = nemo.Simulation(self.nemo_net, self.nemo_conf)

# This switches the default behaviour of Brian's Network object to NemoNetwork
# i.e. if you import this module, then Nemo will be used

def switch_to_nemo_network(self, *args, **kwds):
    self.__class__ = NemoNetwork
    Network_init(self, *args, **kwds)

Network_init = Network.__init__    
Network.__init__ = new.instancemethod(switch_to_nemo_network, None, Network)
