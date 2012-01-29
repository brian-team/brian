from brian import *
from statements import *
from expressions import *
from codeitems import *
from blocks import *
from languages import *
from dependencies import *
from formatting import *

__all__ = [
    'language_invariant_symbol_method',
    'Symbol',
        'RuntimeSymbol',
        'ArraySymbol',
            'NeuronGroupStateVariableSymbol',
        'SliceIndex',
        'ArrayIndex',
    'get_neuron_group_symbols',
    ]


def language_invariant_symbol_method(basemethname, langs, fallback=None):
    def meth(self, *args, **kwds):
        langname = self.language.name
        if langname in langs:
            meth = langs[langname]
            return meth(self, *args, **kwds)
        if fallback is not None:
            return fallback(self, *args, **kwds)
    meth.__name__ = basemethname
    return meth
        
class Symbol(object):
    supported_languages = []
    def __init__(self, name, language):
        self.name = name
        self.language = language
        if not self.supported():
            raise NotImplementedError(
                    "Language "+language.name+" not supported for symbol "+name)
    def supported(self):
        return self.language.name in self.supported_languages
    def update_namespace(self, read, write, vectorisable, namespace):
        pass
    def load(self, read, write, vectorisable):
        return Block()
    def save(self, read, write, vectorisable):
        return Block()
    def read(self):
        return self.name
    def write(self):
        return self.name
    def resolve(self, read, write, vectorisable, item, namespace):
        self.update_namespace(read, write, vectorisable, namespace)
        block = Block(
            self.load(read, write, vectorisable),
            item,
            self.save(read, write, vectorisable))
        block.resolved = block.resolved.union([self.name])
        return block
    def dependencies(self):
        return set()
    def resolution_requires_loop(self):
        return False
    def multi_valued(self):
        if hasattr(self, 'multiple_values'):
            return self.multiple_values
        return False


class RuntimeSymbol(Symbol):
    '''
    This Symbol is guaranteed by the context to be inserted into the namespace
    at runtime and can be used without modification to the name.
    '''
    def supported(self):
        return True


class ArraySymbol(Symbol):
    supported_languages = ['python', 'c']
    def __init__(self, arr, name, language, index=None, array_name=None):
        self.arr = arr
        if index is None:
            index = '_index_'+name
        if array_name is None:
            array_name = '_arr_'+name
        self.index = index
        self.array_name = array_name
        Symbol.__init__(self, name, language)
    # Python implementation
    def load_python(self, read, write, vectorisable):
        if not read:
            return Block()
        code = '{name} = {array_name}[{index}]'.format(
            name=self.name,
            array_name=self.array_name,
            index=self.index)
        dependencies = set([Read(self.index), Read(self.array_name)])
        return CodeStatement(code, dependencies, set())
    def write_python(self):
        return self.array_name+'['+self.index+']'
    # C implementation
    def load_c(self, read, write, vectorisable):
        return CDefineFromArray(self.name, self.array_name,
                                self.index, reference=write, const=(not write),
                                dtype=self.arr.dtype)
    # Language invariant implementation
    def update_namespace(self, read, write, vectorisable, namespace):
        namespace[self.array_name] = self.arr
    def dependencies(self):
        return set([Read(self.index)])
    write = language_invariant_symbol_method('write',
        {'python':write_python}, Symbol.write)
    load = language_invariant_symbol_method('load',
        {'python':load_python, 'c':load_c})


class SliceIndex(Symbol):
    supported_languages = ['python', 'c']
    multiple_values = True
    def __init__(self, name, start, end, language, all=False):
        self.start = start
        self.end = end
        self.all = all
        Symbol.__init__(self, name, language)
    # Python implementation
    def resolve_python(self, read, write, vectorisable, item, namespace):
        if vectorisable:
            if self.all:
                namespace[self.name] = slice(None)
                return item
            code = '{name} = slice({start}, {end})'
            code = code.format(name=self.name, start=self.start, end=self.end)
            return Block(
                CodeStatement(code, self.dependencies(), set()),
                item,
                )
        else:
            container = 'xrange({start}, {end})'.format(
                                                start=self.start, end=self.end)
            return PythonForBlock(self.name, container, item)
    # C implementation
    def resolve_c(self, read, write, vectorisable, item, namespace):
        spec ='int {name}={start}; {name}<{end}; {name}++'.format(
            name=self.name, start=self.start, end=self.end)
        return CForBlock(self.name, spec, item)
    # Language invariant implementation
    def resolution_requires_loop(self):
        return self.language.name!='python'
    resolve = language_invariant_symbol_method('resolve',
        {'c':resolve_c, 'python':resolve_python})

class ArrayIndex(Symbol):
    supported_languages = ['python', 'c']
    multiple_values = True
    def __init__(self, name, array_name, language, array_len=None,
                 index_name=None, array_slice=None):
        if index_name is None:
            index_name = '_index_'+array_name
        if array_len is None:
            array_len = '_len_'+array_name
        self.array_name = array_name
        self.array_len = array_len
        self.index_name = index_name
        self.array_slice = array_slice 
        Symbol.__init__(self, name, language)
    # Python implementation
    def resolve_python(self, read, write, vectorisable, item, namespace):
        if vectorisable:
            code = '{name} = {array_name}'
            start, end = '', ''
            if self.array_slice is not None:
                start, end = self.array_slice
                code += '[{start}:{end}]'
            code = code.format(start=start, end=end,
                name=self.name, array_name=self.array_name)
            block = Block(
                CodeStatement(code, set([Read(self.array_name)]), set()),
                item)
            return block
        else:
            if self.array_slice is None:
                return PythonForBlock(self.name, self.array_name, item)
            else:
                start, end = self.array_slice
                container = '{array_name}[{start}:{end}]'.format(
                    array_name=self.array_name, start=start, end=end)
                return PythonForBlock(self.name, container, item)
    # C implementation
    def resolve_c(self, read, write, vectorisable, item, namespace):
        if self.array_slice is None:
            start, end = '0', self.array_len
        else:
            start, end = self.array_slice
        spec ='int {index_name}={start}; {index_name}<{end}; {index_name}++'
        spec = spec.format(
            index_name=self.index_name, start=start, end=end)
        block = Block(
            CDefineFromArray(self.name, self.array_name, self.index_name,
                             dtype=int, reference=False, const=True),
            item)
        return CForBlock(self.name, spec, block)
    # Language invariant implementation
    def resolution_requires_loop(self):
        return self.language.name!='python'
    resolve = language_invariant_symbol_method('resolve',
        {'c':resolve_c, 'python':resolve_python})


class NeuronGroupStateVariableSymbol(ArraySymbol):
    def __init__(self, group, varname, name, language, index=None):
        self.group = group
        self.varname = varname
        arr = group.state_(varname)
        ArraySymbol.__init__(self, arr, name, language, index=index)

def get_neuron_group_symbols(group, language, index='_neuron_index',
                             subset=False, prefix=''):
    eqs = group._eqs
    symbols = dict(
       (prefix+name,
        NeuronGroupStateVariableSymbol(group, name, prefix+name, language,
               index=index)) for name in eqs._diffeq_names)
    return symbols
