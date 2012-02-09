'''
GPUManager handles:

- Copying memory to/from the GPU
- Holds multiple kernels
- Compiles kernels and returns them
- Combines kernels with the same vectorisation index and range
'''
from brian import *
from codeobject import *
from formatting import *
from languages import *
import statements
try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    import pycuda.compiler as compiler
    from pycuda import gpuarray
except ImportError:
    pycuda = None

__all__ = ['GPUKernel',
           'GPUManager',
           'GPUSymbolMemoryManager',
           'GPUCode',
           'GPULanguage',
           ]


class GPUKernel(object):
    def __init__(self, name, code, namespace, mem_man,
                 maxblocksize=512, scalar='double'):
        self.name = name
        self.code = code
        self.namespace = namespace
        self.mem_man = mem_man
        self.maxblocksize = maxblocksize
        self.scalar = scalar
        self.prepared = False
        self.index = namespace.pop('_gpu_vector_index')
        self.start, self.end = namespace.pop('_gpu_vector_slice')
        self.gpu_func = None
        self._gpu_func_is_prepared = False
    def prepare(self):
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
                if isinstance(value, int):
                    dtype_c = 'int'
                elif isinstance(value, float):
                    dtype_c = self.scalar
                else:
                    continue
                dtype = array(value).dtype.type
                self.func_args.append((name, dtype, dtype_c))
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
        blocksize = self.maxblocksize
        if numinds<blocksize:
            blocksize = numinds
        if numinds%blocksize == 0:
            gridsize = numinds/blocksize
        else:
            gridsize = numinds/blocksize+1
        self.blocksize = blocksize
        self.grid = (gridsize, 1)
    def prepare_gpu_func(self):
        self.gpu_func.prepare(tuple(dtype for _, dtype, _ in self.func_args),
                              (self.blocksize, 1, 1))
        self._gpu_func_is_prepared = True
    def run(self):
        if not self._gpu_func_is_prepared:
            self.prepare_gpu_func()
        ns = self.namespace
        args = [dtype(ns[name]) for name, dtype, _ in self.func_args]
        self.gpu_func.prepared_call(self.grid, *args)


class GPUManager(object):
    def __init__(self):
        self.numkernels = 0
        self.kernels = {}
        self.combined_kernels = []
        self.mem_man = GPUSymbolMemoryManager()
        self.prepared = False
    def add_kernel(self, name, code, namespace):
        basename = name
        suffix = 2
        while name in self.kernels:
            name = basename+str(suffix)
            suffix += 1
        self.kernels[name] = GPUKernel(name, code, namespace, self.mem_man)
        return name
    def make_combined_kernel(self, *names):
        self.combined_kernels.append(names)
    def prepare(self):
        if self.prepared:
            return
        self.prepared = True
        for name, kernel in self.kernels.iteritems():
            kernel.prepare()
        self.generate_code()
        self.compile()
        self.initialise_memory()
    def generate_code(self):
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
        print 'GENERATED GPU SOURCE CODE'
        print code
        self.code = code
    def compile(self):
        self.gpu_mod = compiler.SourceModule(self.code)
        for name, kernel in self.kernels.iteritems():
            kernel.gpu_func = self.gpu_mod.get_function(name)
    def initialise_memory(self):
        for symname in self.mem_man.names:
            fname = self.mem_man.symbol_upload_funcnames[symname]
            f = self.gpu_mod.get_function(fname)
            f(self.mem_man.device[symname], block=(1,1,1))
    def run(self, name):
        kernel = self.kernels[name]
        self.copy_to_device(True) # TODO: temporary
        kernel.run()
        self.copy_to_host(True) # TODO: temporary
        # TODO: combined kernels
    # proxies to mem_man        
    def add_symbols(self, items):
        self.mem_man.add_symbols(items)
    def copy_to_device(self, symname):
        self.mem_man.copy_to_device(symname)
    def copy_to_host(self, symname):
        self.mem_man.copy_to_host(symname)
        

class GPUSymbolMemoryManager(object):
    def __init__(self):
        self.device = {}
        self.host = {}
    def add_symbols(self, items):
#        if pycuda is None:
#            raise ImportError("Cannot import pycuda.")
        for symname, hostarr, devname in items:
            if symname not in self.device:
                if pycuda is not None:
                    devarr = pycuda.gpuarray.to_gpu(hostarr)
                else:
                    devarr = None
                self.device[symname] = devarr
                self.host[symname] = hostarr
    def generate_code(self):
        all_init_arr = ''
        all_arr_ptr = ''
        self.symbol_upload_funcnames = {}
        for name, value in self.host.iteritems():
            if isinstance(value, ndarray):
                dtypestr = statements.c_data_type(value.dtype)
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
        if symname is True:
            self.copy_to_device(self.device.keys())
            return
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_device(name)
            return
        devarr = self.device[symname]
        hostarr = self.host[symname]
        devarr.set(hostarr)
    def copy_to_host(self, symname):
        if symname is True:
            self.copy_to_host(self.device.keys())
            return
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_host(name)
            return
        devarr = self.device[symname]
        hostarr = self.host[symname]
        devarr.get(hostarr)
    @property
    def names(self):
        return self.device.keys()


class GPUCode(Code):
    def __init__(self, name, code_str, namespace, pre_code=None, post_code=None,
                 language=None):
        if language is None:
            raise ValueError("Must provide a language argument to GPUCode.")
        self.gpu_man = language.gpu_man
        name = self.gpu_man.add_kernel(name, code_str, namespace)
        Code.__init__(self, name, code_str, namespace, pre_code=pre_code,
                      post_code=post_code, language=language)
    def compile(self):
        self.gpu_man.prepare()
        self.code_compiled = True
    def run(self):
        self.gpu_man.run(self.name)


class GPULanguage(CLanguage):
    CodeObjectClass = GPUCode
    def __init__(self, scalar='double', gpu_man=None):
        Language.__init__(self, 'gpu')
        self.scalar = scalar
        if gpu_man is None:
            gpu_man = GPUManager()
        self.gpu_man = gpu_man
