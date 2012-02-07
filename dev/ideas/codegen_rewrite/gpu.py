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

__all__ = ['GPUSymbolMemoryManager',
           'GPUCode',
           'GPULanguage',
           ]
        

class GPUSymbolMemoryManager(object):
    def __init__(self):
        self.device = {}
        self.host = {}
    def add_symbols(self, items):
        if pycuda is None:
            raise ImportError("Cannot import pycuda.")
        for symname, hostarr, devname in items:
            if symname not in self.device:
                devarr = pycuda.gpuarray.to_gpu(hostarr)
                self.device[symname] = devarr
                self.host[symname] = hostarr
    def copy_to_device(self, symname):
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_device(name)
            return
        devarr = self.device[symname]
        hostarr = self.host[symname]
        devarr.set(hostarr)
    def copy_to_host(self, symname):
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_host(name)
            return
        devarr = self.device[symname]
        hostarr = self.host[symname]
        devarr.get(hostarr)


class GPUCode(Code):
    def __init__(self, code_str, namespace, pre_code=None, post_code=None,
                 language=None, maxblocksize=512, forcesync=True):
        vector_index = namespace.pop('_gpu_vector_index')
        start, end = namespace.pop('_gpu_vector_slice')
        code_str_template = '''
        {{
            int {vector_index} = blockIdx.x * blockDim.x + threadIdx.x;
            if(({vector_index}<({start}))||({vector_index}>=({end})))
                return;
        {code_str}
        }}
        '''
        code_str = flattened_docstring(code_str_template).format(
            vector_index=vector_index, start=start, end=end,
            code_str=indent_string(code_str))
        code_str = strip_empty_lines(code_str)
        # TODO: TEMPORARY HACK
        self.scalar = 'double'
        self.maxblocksize = maxblocksize
        self.forcesync = forcesync
        Code.__init__(self, code_str, namespace, pre_code=pre_code,
                      post_code=post_code, language=language)
        if language is None:
            mem_man = GPUSymbolMemoryManager()
        else:
            mem_man = language.mem_man
        self.mem_man = mem_man
    def compile(self):
        ns = self.namespace
        numinds = ns.pop('_num_gpu_indices')
        all_init_arr = ''
        all_arr_ptr = ''
        self.func_args = []
        memman_items = []
        for name, value in ns.iteritems():
            if isinstance(value, ndarray):
                dtypestr = statements.c_data_type(value.dtype)
                init_arr_template = '''
                __global__ void set_array_{name}({dtypestr} *_{name})
                {{
                    {name} = _{name};
                }}
                ''' 
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
                # TODO: we should have access to the symbol name so that the
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
        self.code_str = '__global__ void gpu_func({funcargs})\n'.format(
            funcargs=', '.join(dtype_c+' '+name for name, dtype, dtype_c in self.func_args))+self.code_str     
        self.code_str = all_arr_ptr+all_init_arr+self.code_str
        print 'GPU modified code:'
        print self.code_str
        print 'GPU namespace keys:'
        print ns.keys()
        self.mem_man.add_symbols(memman_items)
        self.gpu_mod = compiler.SourceModule(self.code_str)
        self.gpu_func = self.gpu_mod.get_function("gpu_func")
        for symname, hostarr, devname in memman_items:
            f = self.gpu_mod.get_function("set_array_"+devname)
            f(self.mem_man.device[symname], block=(1,1,1))
        self.code_compiled = True
        blocksize = self.maxblocksize
        if numinds<blocksize:
            blocksize = numinds
        if numinds%blocksize == 0:
            gridsize = numinds/blocksize
        else:
            gridsize = numinds/blocksize+1
        self.gpu_func.prepare(tuple(dtype for _, dtype, _ in self.func_args),
                              (blocksize, 1, 1))
        self._gpu_grid = (gridsize, 1)
    def run(self):
        ns = self.namespace
        args = [dtype(ns[name]) for name, dtype, _ in self.func_args]
        if self.forcesync:
            self.mem_man.copy_to_device(self.mem_man.device.keys())
        self.gpu_func.prepared_call(self._gpu_grid, *args)
        if self.forcesync:
            self.mem_man.copy_to_host(self.mem_man.device.keys())


class GPULanguage(CLanguage):
    CodeObjectClass = GPUCode
    def __init__(self, scalar='double', mem_man=None):
        Language.__init__(self, 'gpu')
        self.scalar = scalar
        if mem_man is None:
            mem_man = GPUSymbolMemoryManager()
        self.mem_man = mem_man
