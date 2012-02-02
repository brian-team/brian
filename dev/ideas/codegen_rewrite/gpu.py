from brian import *
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
           ]


class GPUSymbolMemoryManager(object):
    def __init__(self, items):
        self.device = {}
        self.host = {}
        for symname, hostarr, devname in items:
            devarr = pycuda.gpuarray.to_gpu(hostarr)
            self.device[symname] = devarr
            self.host[symname] = hostarr
    def copy_to_device(self, symname):
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_device(name)
        devarr = self.device[symname]
        hostarr = self.host[symname]
        devarr.get(hostarr)
    def copy_to_host(self, symname):
        if not isinstance(symname, str):
            for name in symname:
                self.copy_to_host(name)
        devarr = self.device[symname]
        hostarr = self.host[symname]
        devarr.set(hostarr)


class GPUCode(Code):
    def __init__(self, code_str, namespace, pre_code=None, post_code=None,
                 maxblocksize=512):
        if pycuda is None:
            raise ImportError("Cannot import pycuda.")
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
        Code.__init__(self, code_str, namespace, pre_code=pre_code,
                      post_code=post_code)
    def compile(self):
        ns = self.namespace
        all_init_arr = ''
        all_arr_ptr = ''
        self.func_args = []
        memman_items = []
        for name, value in ns.iteritems():
            if isinstance(value, ndarray):
                init_arr_template = '''
                __global__ void set_array_{name}({scalar} *_{name})
                {{
                    {name} = _{name};
                }}
                ''' 
                # TODO: doens't have to be scalar type?
                init_arr = flattened_docstring(init_arr_template).format(
                                                scalar=self.scalar, name=name)
                arr_ptr_template = '''
                __const__ {scalar} *{name};
                '''
                arr_ptr = flattened_docstring(arr_ptr_template).format(
                                                scalar=self.scalar, name=name)
                all_init_arr += init_arr
                all_arr_ptr += arr_ptr
                # TODO: we should have access to the symbol name so that the
                # memory manager can be used by the user to get/set values
                symname = devname = name
                hostarr = value
                memman_items.append((symname, hostarr, devname))
            else:
                if isinstance(value, int):
                    dtype = 'int'
                elif isinstance(value, float):
                    dtype = self.scalar
                else:
                    continue
                self.func_args.append((name, dtype))
        self.code_str = '__global__ void gpu_func({funcargs})\n'.format(
            funcargs=', '.join(dtype+' '+name for name, dtype in self.func_args))+self.code_str
        self.code_str = all_arr_ptr+all_init_arr+self.code_str
        print 'GPU modified code:'
        print self.code_str
        print 'GPU namespace keys:'
        print ns.keys()
        self.mem_man = GPUSymbolMemoryManager(memman_items)
        self.gpu_mod = compiler.SourceModule(self.code_str)
        self.gpu_func = self.gpu_mod.get_function("gpu_func")
        for symname, hostarr, devname in memman_items:
            f = self.gpu_mod.get_function("set_array_"+devname)
            f(self.mem_man.device[symname])
        self.code_compiled = True
#    def _prepare(self, P):
#        blocksize = self.maxblocksize
#        if len(P) < blocksize:
#            blocksize = len(P)
#        if len(P) % blocksize == 0:
#            gridsize = len(P) / blocksize
#        else:
#            gridsize = len(P) / blocksize + 1
#        self._prepared = True
#        self.gpu_func.prepare((int32, self.precision_dtype, 'i'), (blocksize, 1, 1))
#        self._S_gpu_addr = P._S.gpu_pointer
#        self._gpu_N = int32(len(P))
#        self._gpu_grid = (gridsize, 1)
#    def __call__(self, P):
#        if not self._prepared:
#            self._prepare(P)
#        P._S.sync_to_gpu()
#        self.gpu_func.prepared_call(self._gpu_grid, self._gpu_N, self.precision_dtype(P.clock._t), self._S_gpu_addr)
#        P._S.changed_gpu_data()
#        if self.forcesync:
#            P._S.sync_to_cpu()
#            P._S.changed_cpu_data()
    def run(self):
        pass
