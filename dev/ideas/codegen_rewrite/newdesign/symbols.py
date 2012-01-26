from brian import *
from statements import *
from expressions import *
from codeitems import *
from blocks import *
from languages import *
from dependencies import *
from formatting import *

__all__ = [
    'Symbol',
        'RuntimeSymbol',
        'ArraySymbol',
            'NeuronGroupStateVariableSymbol',
        'IndexSymbol',
        'get_neuron_group_symbols',
    ]

class Symbol(object):
    def __init__(self, name, language):
        self.name = name
        self.language = language
        if not self.supported():
            raise NotImplementedError(
                    "Language "+language.name+" not supported for symbol "+name)
    def supported():
        return False
    def update_namespace(self, read, write, namespace):
        pass
    def load(self, read, write):
        return Block()
    def save(self, read, write):
        return Block()
    def read(self):
        return self.name
    def write(self):
        return self.name
    def resolve(self, read, write, item, namespace):
        self.update_namespace(read, write, namespace)
        block = Block(
            self.load(read, write),
            item,
            self.save(read, write))
        block.resolved = block.resolved.union([self.name])
        return block
    def dependencies(self):
        return set()
    def resolution_requires_loop(self):
        return False


class RuntimeSymbol(Symbol):
    '''
    This Symbol is guaranteed by the context to be inserted into the namespace
    at runtime and can be used without modification to the name.
    '''
    def supported(self):
        return True


class ArraySymbol(Symbol):
    def __init__(self, arr, name, language, index=None, subset=False,
                 array_name=None, readname=None):
        self.arr = arr
        if index is None:
            index = '_index_'+name
        if array_name is None:
            if language.name=='c':
                array_name = '_arr_'+name
            elif language.name=='python':
                array_name = name
        if readname is None:
            readname = '_read_'+name
        self.index = index
        self.subset = subset
        self.array_name = array_name
        self.readname = readname
        Symbol.__init__(self, name, language)
    def supported(self):
        return self.language.name in ['python', 'c']
    def update_namespace(self, read, write, namespace):
        if read or write:
            langname = self.language.name
            if langname=='python':
                namespace[self.array_name] = self.arr
            elif langname=='c':
                namespace[self.array_name] = self.arr
    def load(self, read, write):
        langname = self.language.name
        if langname=='python':
            if not self.subset or not read:
                block = Block()
                if self.subset:
                    block.dependencies = set([Read(self.index)]) 
                return block
            read_name = self.read()
            code = '{read_name} = {array_name}[{index}]'.format(
                read_name=read_name,
                array_name=self.array_name,
                index=self.index)
            dependencies = set([Read(self.index), Read(self.array_name)])
            resolved = set([read_name])
            return CodeStatement(code, dependencies, resolved)
        elif langname=='c':
            return CDefineFromArray(self.name, self.array_name,
                                    self.index, reference=write,
                                    dtype=self.arr.dtype)
    def read(self):
        langname = self.language.name
        if langname=='python':
            if self.subset:
                return self.readname
            else:
                return self.array_name
        elif langname=='c':
            return self.name
    def write(self):
        langname = self.language.name
        if langname=='python':
            writename = self.array_name
            if self.subset:
                writename += '['+self.index+']'
            else:
                writename += '[:]'
            return writename
        elif langname=='c':
            return self.name
    def dependencies(self):
        if self.language.name=='python' and not self.subset:
            return set()
        else:
            return set([Read(self.index)])
    def resolution_requires_loop(self):
        return False


class IndexSymbol(Symbol):
    def __init__(self, name, start, end, language, index_array=None,
                 forced_dependencies=None):
        if forced_dependencies is None:
            forced_dependencies = set()
        self.index_array = index_array
        self.start = start
        self.end = end
        self.forced_dependencies = forced_dependencies
        Symbol.__init__(self, name, language)
    def supported(self):
        return self.language.name in ['python', 'c']
    def resolve(self, read, write, item, namespace):
        langname = self.language.name
        if langname=='python':
            if self.index_array is None:
                code = '{name} = slice({start}, {end})'.format(
                                name=self.name, start=self.start, end=self.end)
            else:
                code = '{name} = {index_array}'.format(name=self.name,
                                                index_array=self.index_array)
            return Block(
                CodeStatement(code, self.dependencies(), set()),
                item,
                )
        elif langname=='c':
            if self.index_array is None:
                for_index = self.name
                content = item
            else:
                for_index = '_index_'+self.index_array
                content = Block(
                    CDefineFromArray(self.name, self.index_array, for_index,
                                     dtype=int, reference=False),
                    item
                    )
            spec = 'int {i}={start}; {i}<{end}; {i}++'.format(
                                    i=for_index, start=self.start, end=self.end)
            return CForBlock(for_index, spec, content)
    def dependencies(self):
        deps = set()
        if self.language.name=='c' and self.index_array is not None:
            deps.update(set([Read(self.index_array)]))
        deps.update(set(Read(x) for x in get_identifiers(self.start)))
        deps.update(set(Read(x) for x in get_identifiers(self.end)))
        deps.update(self.forced_dependencies)
        return deps
    def resolution_requires_loop(self):
        return self.language.name=='c'


class NeuronGroupStateVariableSymbol(ArraySymbol):
    def __init__(self, group, varname, name, language,
                 index=None, subset=False):
        self.group = group
        self.varname = varname
        arr = group.state_(varname)
        ArraySymbol.__init__(self, arr, name, language, index=index,
                             subset=subset)

def get_neuron_group_symbols(group, language, index='_neuron_index',
                             subset=False, prefix=''):
    eqs = group._eqs
    symbols = dict(
       (prefix+name,
        NeuronGroupStateVariableSymbol(group, name, prefix+name, language,
               index=index, subset=subset)) for name in eqs._diffeq_names)
    return symbols
