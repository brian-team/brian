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
    def __init__(self, arr, name, language, index=None, subset=False):
        self.arr = arr
        if index is None:
            index = '_index_'+name
        self.index = index
        self.subset = subset
        Symbol.__init__(self, name, language)
    def supported(self):
        return self.language.name in ['python', 'c']
    def update_namespace(self, read, write, namespace):
        if read or write:
            langname = self.language.name
            if langname=='python':
                namespace[self.name] = self.arr
            elif langname=='c':
                namespace['_arr_'+self.name] = self.arr
    def load(self, read, write):
        langname = self.language.name
        if langname=='python':
            if not self.subset:
                return Block()
            read_name = self.read()
            code = '{read_name} = {name}[{index}]'.format(
                read_name=read_name,
                name=self.name,
                index=self.index)
            dependencies = set([Read(self.index)])
            resolved = set([read_name])
            return CodeStatement(code, dependencies, resolved)
        elif langname=='c':
            return CDefineFromArray(self.name, '_arr_'+self.name,
                                    self.index, reference=write,
                                    dtype=self.arr.dtype)
    def read(self):
        langname = self.language.name
        if langname=='python':
            if self.subset:
                return '_read_'+self.name
            else:
                return self.name
        elif langname=='c':
            return self.name
    def write(self):
        langname = self.language.name
        if langname=='python':
            writename = self.name
            if self.subset:
                writename += '['+self.index_array+']'
            else:
                writename += '[:]'
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
    def __init__(self, name, N, language, index_array=None):
        self.index_array = index_array
        self.N = N
        Symbol.__init__(self, name, language)
    def supported(self):
        return self.language.name in ['python', 'c']
    def resolve(self, read, write, item, namespace):
        langname = self.language.name
        if langname=='python':
            return item
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
            spec = '{i}=0; {i}<{N}; {i}++'.format(i=for_index, N=self.N)
            return CForBlock(for_index, spec, content)
    def dependencies(self):
        if self.language.name=='c' and self.index_array is not None:
            return set(Read(self.index_array))
        return set()
    def resolution_requires_loop(self):
        return self.language.name=='c'


class NeuronGroupStateVariableSymbol(ArraySymbol):
    def __init__(self, group, varname, name, language, index=None, subset=False):
        self.group = group
        self.varname = varname
        arr = group._var(varname)
        ArraySymbol.__init__(self, arr, name, language, index=index,
                             subset=subset)

if __name__=='__main__':
    item = MathematicalStatement('x', ':=', 'y*y')
    y = zeros(10)
    #idx = array([1, 3, 5])
    subset = True
    language = PythonLanguage()
    #language = CLanguage()
    sym_y = ArraySymbol(y, 'y', language,
                      index='idx',
                      subset=subset,
                      )
    if subset:
        N = 'idx_arr_len'
        sym_idx = IndexSymbol('idx', N, language, index_array='idx_arr')
    else:
        N = 'y_len'
        sym_idx = IndexSymbol('idx', N, language)
    symbols = {'y':sym_y, 'idx':sym_idx}
    print 'Code:\n', indent_string(item.convert_to(language, symbols)),
    print 'Dependencies:', item.dependencies
    print 'Resolved:', item.resolved
    print
    read = Read('y') in item.dependencies
    write = Write('y') in item.dependencies
    namespace = {}
    item = sym_y.resolve(read, write, item, namespace)
    print 'Code:\n', indent_string(item.convert_to(language, symbols)),
    print 'Dependencies:', item.dependencies
    print 'Resolved:', item.resolved
    print
    read = Read('idx') in item.dependencies
    write = Write('idx') in item.dependencies
    item = sym_idx.resolve(read, write, item, namespace)
    print 'Code:\n', indent_string(item.convert_to(language, symbols)),
    print 'Dependencies:', item.dependencies
    print 'Resolved:', item.resolved
    print
    print 'Namespace:', namespace.keys()
