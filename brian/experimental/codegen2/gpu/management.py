'''
GPUManager handles:

- Copying memory to/from the GPU
- Holds multiple kernels
- Compiles kernels and returns them
- Combines kernels with the same vectorisation index and range
'''
from brian import *
from ..codeobject import *
from ..formatting import *
from ..languages import *
from importpycuda import *
import re

__all__ = ['GPUKernel',
           'GPUManager',
           'GPUSymbolMemoryManager',
           'GPUCode',
           'GPULanguage',
           'compute_block_grid_size', 'compute_block_grid',
           ]

def compute_block_grid_size(maxblocksize, numinds):
    blocksize = maxblocksize
    if numinds<blocksize:
        blocksize = numinds
    if numinds%blocksize==0:
        gridsize = numinds/blocksize
    else:
        gridsize = numinds/blocksize+1
    return int(blocksize), int(gridsize)

def compute_block_grid(maxblocksize, numinds):
    blocksize, gridsize = compute_block_grid_size(maxblocksize, numinds)
    return (blocksize, 1, 1), (gridsize, 1)

class GPUKernel(object):
    '''
    Generates final kernel source code and used to launch kernels.
    
    Used in conjunction with :class:`GPUManager`. Each kernel is prepared with
    :meth:`prepare` which generates source code and adds symbols to the
    :class:`GPUSymbolMemoryManager`. The :class:`GPUManager` compiles the
    code and sets the :attr:`gpu_func` attribute, and the kernel can then be
    called via :meth:`run`.
    
    The initialisation method extracts variable ``_gpu_vector_index`` from the
    namespace and stores it as attribute ``index``, and ``_gpu_vector_slice``
    as the pair ``(start, end)``.
    '''
    def __init__(self, name, code, namespace, mem_man,
                 maxblocksize=512, scalar='double', force_sync=True):
        self.name = name
        self.code = code
        self.namespace = namespace
        self.mem_man = mem_man
        self.maxblocksize = maxblocksize
        self.scalar = scalar
        self.force_sync = force_sync
        self.prepared = False
        self.index = namespace.pop('_gpu_vector_index')
        self.start, self.end = namespace.pop('_gpu_vector_slice')
        self.gpu_func = None
        self._gpu_func_is_prepared = False
    def prepare(self):
        '''
        Generates kernel source code and adds symbols to memory manager.
        
        We extract the number of GPU indices from the namespace,
        ``_num_gpu_indices``.
        
        We loop through the namespace, and for each value determine it to be
        either an array or a single value. If it is an array, then we place
        it in the :class:`GPUSymbolMemoryManager`, otherwise we add it to the
        list of arguments provided to the function call. This allows scalar
        variables like ``t`` to be transmitted to the kernel function in its
        arguments.
        
        We then generate a kernel of the following (Python template) form::
        
            __global__ void {name}({funcargs})
            {{
                int {vector_index} = blockIdx.x * blockDim.x + threadIdx.x;
                if(({vector_index}<({start}))||({vector_index}>=({end})))
                    return;
            {code_str}
            }}
        
        We also compute the block size and grid size using the user provided
        maximum block size.
        '''
        from ..statements import c_data_type
        if self.prepared:
            return
        self.prepared = True
        ns = self.namespace
        code_str = self.code
        self.numinds = numinds = ns.pop('_num_gpu_indices')
        self.func_args = []
        memman_items = []
        for name, value in ns.iteritems():
            if isinstance(value, ndarray):
                # TODO: we shouldself._gpu_func_is_prepared have access to the symbol name so that the
                # memory manager can be used by the user to get/set values
                symname = devname = name
                hostarr = value
                memman_items.append((symname, hostarr, devname))
            else:
                dtype = array(value).dtype.type
                if dtype==float64 and self.scalar=='float':
                    dtype = float32
                dtype_c = c_data_type(dtype)
                self.func_args.append((name, dtype, dtype_c))
        # This allocates memory on the GPU
        self.mem_man.add_symbols(memman_items)
        func_args_str = ', '.join(dtype_c+' '+name for name, _, dtype_c in self.func_args)
        code_str_template = '''
        __global__ void {name}({funcargs})
        {{
            int {vector_index} = blockIdx.x * blockDim.x + threadIdx.x;
            if(({vector_index}<({start}))||({vector_index}>=({end})))
                return;
        {code_str}
        }}
        '''
        code_str = flattened_docstring(code_str_template).format(
            vector_index=self.index, start=self.start, end=self.end,
            code_str=indent_string(code_str),
            name=self.name, funcargs=func_args_str)
        self.kernel_code = code_str
        self.blocksize, self.gridsize = compute_block_grid_size(self.maxblocksize, numinds)
        self.grid = (self.gridsize, 1)
    def prepare_gpu_func(self):
        '''
        Calls the ``pycuda`` GPU function ``prepare()`` method for low-overhead
        function calls.
        '''
        self.gpu_func.prepare(tuple(dtype for _, dtype, _ in self.func_args),
                              (self.blocksize, 1, 1))
        self._gpu_func_is_prepared = True
    def run(self):
        '''
        Calls the function on the GPU, extracting the scalar variables in the
        argument list from the namespace.
        '''
        if not self._gpu_func_is_prepared:
            self.prepare_gpu_func()
        ns = self.namespace
        args = [dtype(ns[name]) for name, dtype, _ in self.func_args]
#        print self.gpu_func.local_size_bytes, self.gpu_func.shared_size_bytes, self.gpu_func.num_regs
#        print self.grid, self.blocksize
        self.gpu_func.prepared_call(self.grid, *args)


class GPUManager(object):
    '''
    This object controls everything on the GPU.
    
    It uses a :class:`GPUKernel` object for managing kernels, and a
    :class:`GPUSymbolMemoryManager` object for managing symbol memory.
    
    The class is used by:
    
    1. Adding several kernels using :meth:`add_kernel`
    2. Calling :meth:`prepare` (see method documentation for details)
    3. Run code with :meth:`run`
    
    Memory is mirrored on GPU and CPU. In the present implementation, in the
    development phase only, each call to :meth:`run` will copy all symbols from
    CPU to GPU before running the GPU kernel, and afterwards copy all symbols
    from GPU back to CPU. In the future, this will be disabled and symbol
    memory copies will be handled explicitly by calls to methods
    :meth:`copy_to_device` and :meth:`copy_to_host`.
    '''
    def __init__(self, force_sync=True, usefloat=False):
        self.numkernels = 0
        self.kernels = {}
        self.combined_kernels = []
        self.mem_man = GPUSymbolMemoryManager(usefloat=usefloat)
        self.prepared = False
        self.force_sync = force_sync
        self.usefloat = usefloat
    def add_kernel(self, name, code, namespace):
        '''
        Adds a kernel with the given name, code and namespace. Creates a
        :class:`GPUKernel` object.
        '''
        basename = name
        suffix = 2
        while name in self.kernels:
            name = basename+str(suffix)
            suffix += 1
        if self.usefloat:
            scalar = 'float'
        else:
            scalar = 'double'
        self.kernels[name] = GPUKernel(name, code, namespace, self.mem_man,
                                       force_sync=self.force_sync,
                                       scalar=scalar)
        return name
    def make_combined_kernel(self, *names):
        '''
        Not used at present. Will be used to combine multiple kernels with
        the same vectorisation index for efficiency.
        '''
        self.combined_kernels.append(names)
    def prepare(self):
        '''
        Compiles code and initialises memory.
        
        Performs the following steps:
        
        1. :meth:`GPUKernel.prepare` is called for each kernel, converting the
           partial code into a complete kernel, and adds symbols to the
           :class:`GPUSymbolMemoryManager`, which allocates space on the GPU and
           copies data to it from the CPU.
        2. :meth:`generate_code` is called, combining individual kernels into
           one source file, and adding memory management kernels and
           declarations.
        3. :meth:`compile` is called, which JIT-compiles the code using
           ``pycuda``.
        4. :meth:`initialise_memory` is called, which allocates memory 
        '''
        if self.prepared:
            return
        self.prepared = True
        for name, kernel in self.kernels.iteritems():
            kernel.prepare()
        self.generate_code()
        self.compile()
        self.initialise_memory()
    def generate_code(self):
        '''
        Combines kernel source into one source file, and adds memory management
        kernel functions. These simple kernels simply copy a pointer to a
        previously specified name. This is necessary because when ``pycuda`` is
        used to allocate memory, it doesn't give it a name only a pointer, and
        the kernel functions use a named array.
        
        Calls :meth:`GPUSymbolMemoryManager.generate_code`.
        '''
        self.prepare()
        code = ''
        # global memory arrays
        code += self.mem_man.generate_code()
        # individual kernels
        for name, kernel in self.kernels.iteritems():
            code += kernel.kernel_code
        # combined kernels
#        for names in self.combined_kernels:
#            combined_name = 'combined_'+'_'.join(names)
#            for name in names:
#                code, ns, index, _ = self.kernels[name]
        # if necessary, convert double to float
        if self.usefloat:
            code = re.sub(r'\bdouble\b', 'float', code)
        log_info('brian.codegen2.gpu.GPUManager', 'GENERATED GPU SOURCE CODE:\n'+code)
        self.code = code
    def compile(self):
        '''
        Compiles code using :class:`pycuda.compiler.SourceModule` and extracts
        kernel functions with :meth:`pycuda.compiler.SourceModule.get_function`.
        The :attr:`GPUKernel.gpu_func` attribute is set for each kernel.
        '''
        self.gpu_mod = compiler.SourceModule(self.code)
        for name, kernel in self.kernels.iteritems():
            kernel.gpu_func = self.gpu_mod.get_function(name)
    def initialise_memory(self):
        '''
        Copies allocated memory pointers to named global memory pointer
        variables so that kernels can use them. The kernel names to do this
        are in the :attr:`GPUSymbolMemoryManager.symbol_upload_funcnames`
        dict (keys are symbol names), and the allocated pointers are in
        the :attr:`GPUSymbolMemoryManager.device` dict.
        '''
        for symname in self.mem_man.names:
            fname = self.mem_man.symbol_upload_funcnames[symname]
            f = self.gpu_mod.get_function(fname)
            f(self.mem_man.device[symname], block=(1,1,1))
    def run(self, name):
        '''
        Runs the named kernel. Calls :meth:`GPUKernel.run`. Note that all
        symbols are copied to and from the GPU before and after the kernel run,
        although this is only for the development phase and will change later.
        '''
        kernel = self.kernels[name]
        if self.force_sync:
            self.copy_to_device(True) # TODO: temporary
        kernel.run()
        if self.force_sync:
            self.copy_to_host(True) # TODO: temporary
        # TODO: combined kernels
    # proxies to mem_man        
    def add_symbols(self, items):
        '''
        Proxy to :meth:`GPUSymbolMemoryManager.add_symbols`.
        '''
        self.mem_man.add_symbols(items)
    def copy_to_device(self, symname):
        '''
        Proxy to :meth:`GPUSymbolMemoryManager.copy_to_device`.
        '''
        self.mem_man.copy_to_device(symname)
    def copy_to_host(self, symname):
        '''
        Proxy to :meth:`GPUSymbolMemoryManager.copy_to_host`.
        '''
        self.mem_man.copy_to_host(symname)
        

class GPUSymbolMemoryManager(object):
    '''
    Manages symbol memory on the GPU.
    
    Stores an attribute ``device`` and ``host`` which are dicts, with keys the
    symbol names, and values :class:`pycuda.gpuarray.GPUArray` and
    :class:`numpy.ndarray` respectively. Add symbols with :meth:`add_symbols`,
    which will allocate memory.
    '''
    def __init__(self, usefloat=False):
        self.device = {}
        self.host = {}
        self.usefloat = usefloat
    def add_symbols(self, items):
        '''
        Adds a collection of symbols.
        
        Each item in ``items`` is of the form ``(symname, hostarr, devname)``
        where ``symname`` is the symbol name, ``hostarr`` is the
        :class:`numpy.ndarray` containing the data, and ``devname`` is the name
        the array pointer should have on the device.
        
        Allocates memory on the device, and copies data to the GPU.
        '''
#        if pycuda is None:
#            raise ImportError("Cannot import pycuda.")
        for symname, hostarr, devname in items:
            if symname not in self.device:
                if pycuda is not None:
                    if self.usefloat and hostarr.dtype==float64:
                        devarr = pycuda.gpuarray.to_gpu(array(hostarr, dtype=float32))
                    else:
                        devarr = pycuda.gpuarray.to_gpu(hostarr)
                else:
                    devarr = None
                self.device[symname] = devarr
                self.host[symname] = hostarr
    def generate_code(self):
        '''
        Generates declarations for array pointer names on the device, and
        kernels to copy device pointers to the array pointers. General form
        is::
        
            __device__ {dtypestr} *{name};
            __global__ void set_array_{name}({dtypestr} *_{name})
            {
                {name} = _{name};
            }
            
        Stores the kernel function names in attribute
        :attr:`symbol_upload_funcnames` (dict with keys being symbol names).
        
        Returns a string with declarations and kernels combined.
        '''
        all_init_arr = ''
        all_arr_ptr = ''
        self.symbol_upload_funcnames = {}
        for name, value in self.host.iteritems():
            if isinstance(value, ndarray):
                from ..statements import c_data_type
                dtypestr = c_data_type(value.dtype)
                if self.usefloat and dtypestr=='double':
                    dtypestr = 'float'
                init_arr_template = '''
                __global__ void set_array_{name}({dtypestr} *_{name})
                {{
                    {name} = _{name};
                }}
                ''' 
                self.symbol_upload_funcnames[name] = 'set_array_'+name
                # TODO: handle double/float
                init_arr = flattened_docstring(init_arr_template).format(
                                                dtypestr=dtypestr, name=name)
                arr_ptr_template = '''
                __device__ {dtypestr} *{name};
                '''
                arr_ptr = flattened_docstring(arr_ptr_template).format(
                                                dtypestr=dtypestr, name=name)
                all_init_arr += init_arr
                all_arr_ptr += arr_ptr
        return all_arr_ptr+all_init_arr
    def copy_to_device(self, symname):
        '''
        Copy the memory in the :class:`numpy.ndarray` for ``symname`` to the
        allocated device memory. If ``symname==True``, do this for all symbols.
        You can also pass a list for ``symname``.
        '''
        if symname is True:
            self.copy_to_device(self.device.keys())
            return
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_device(name)
            return
        devarr = self.device[symname]
        hostarr = self.host[symname]
        if self.usefloat and hostarr.dtype==float64:
            # TODO: make this pagelocked/cached for speed?
            hostarr = array(hostarr, dtype=float32)
        devarr.set(hostarr)
    def copy_to_host(self, symname):
        '''
        As for :meth:`copy_to_device` but copies memory from device to host.
        '''
        if symname is True:
            self.copy_to_host(self.device.keys())
            return
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_host(name)
            return
        devarr = self.device[symname]
        hostarr = self.host[symname]
        if devarr.dtype==float32 and hostarr.dtype==float64:
            # TODO: cache the float32 intermediate array for speed
            #hostarr[:] = array(devarr.get(pagelocked=True), dtype=float64)
            hostarr[:] = devarr.get(pagelocked=True)
        else:
            devarr.get(hostarr)
    @property
    def names(self):
        '''
        The list of symbol names managed.
        '''
        return self.device.keys()


class GPUCode(Code):
    '''
    :class:`Code` object for GPU.
    
    For the user, works as the same as any other :class:`Code` object. Behind
    the scenes, source code is passed to the :class:`GPUManager` ``gpu_man``
    from the :class:`GPULanguage` object, via :meth:`GPUManager.add_kernel`.
    Compilation is handled by :meth:`GPUManager.prepare`, and running code
    by :meth:`GPUManager.run`.
    '''
    def __init__(self, name, code_str, namespace, pre_code=None, post_code=None,
                 language=None):
        if language is None:
            raise ValueError("Must provide a language argument to GPUCode.")
        self.gpu_man = language.gpu_man
        name = self.gpu_man.add_kernel(name, code_str, namespace)
        Code.__init__(self, name, code_str, namespace, pre_code=pre_code,
                      post_code=post_code, language=language)
    def compile(self):
        '''
        Simply calls :meth:`GPUManager.prepare`.
        '''
        self.gpu_man.prepare()
        self.code_compiled = True
    def run(self):
        '''
        Simply runs the kernel via :meth:`GPUManager.run`.
        '''
        self.gpu_man.run(self.name)


class GPULanguage(CLanguage):
    '''
    :class:`Language` object for GPU.
    
    Has an attribute ``gpu_man``, the :class:`GPUManager` object responsible for
    allocating, copying memory, etc. One is created if you do not specify one.
    '''
    CodeObjectClass = GPUCode
    def __init__(self, scalar='double', gpu_man=None, force_sync=True):
        Language.__init__(self, 'gpu')
        self.scalar = scalar
        if gpu_man is None:
            gpu_man = GPUManager(force_sync=force_sync,
                                 usefloat=(scalar=='float'))
        self.gpu_man = gpu_man
        self.force_sync = force_sync
