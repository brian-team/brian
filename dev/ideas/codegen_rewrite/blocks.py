from brian import *
from statements import *
from dependencies import *
from codeitems import *
from languages import *
from formatting import *

__all__ = [
    'Block',
    'ControlBlock',
        'ForBlock',
            'PythonForBlock',
            'CForBlock',
        'IfBlock',
            'PythonIfBlock',
            'CIfBlock',
    ]

class Block(CodeItem):
    def __init__(self, *args):
        self.contents = list(args)
    def __iter__(self):
        return iter(self.contents)

class ControlBlock(Block):
    def __init__(self, start, end, contents, dependencies, resolved):
        self.selfdependencies = dependencies
        self.selfresolved = resolved
        self.start = start
        self.end = end
        if isinstance(contents, CodeItem):
            contents = [contents]
        self.contents = contents
    def convert_to(self, language, symbols={}):
        contentcode = Block.convert_to(self, language, symbols=symbols)
        s = self.start+'\n'+indent_string(contentcode)+'\n'+self.end
        return strip_empty_lines(s)

class ForBlock(ControlBlock):
    pass

class PythonForBlock(ForBlock):
    def __init__(self, var, container, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set([Read(x) for x in get_identifiers(container)])
        if resolved is None:
            resolved = set([var])
        self.var = var
        self.container = container
        start = 'for {var} in {container}:'.format(var=var, container=container)
        end = ''
        ControlBlock.__init__(self, start, end, content, dependencies, resolved)

class CForBlock(ForBlock):
    def __init__(self, var, spec, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set([Read(x) for x in get_identifiers(spec)])
            dependencies.discard(Read(var))
        if resolved is None:
            resolved = set([var])
        start = 'for({spec})\n{{'.format(spec=spec)
        end = '}'
        ControlBlock.__init__(self, start, end, content, dependencies, resolved)

class CIterateArray(CForBlock):
    def __init__(self, var, arr, arr_len, content,
                 index=None,
                 dependencies=None, resolved=None,
                 dtype=None, reference=True):
        if index is None:
            index = '_index_'+var
        if dependencies is None:
            dependencies = get_identifiers(arr)+[arr_len]
            dependencies = set([Read(x) for x in dependencies])
        if resolved is None:
            resolved = set([var, index])
        spec = 'int {index}=0; {index}<{arr_len}; {index}++'.format(
                                                index=index, arr_len=arr_len)
        def_var = CDefineFromArray(var, arr, index,
                                   dtype=dtype, reference=reference)
        content = [def_var, Block(content)]
        CForBlock.__init__(self, var, spec, content,
                           dependencies=dependencies, resolved=resolved)

class IfBlock(ControlBlock):
    pass

class PythonIfBlock(IfBlock):
    def __init__(self, cond, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set(get_identifiers(cond))
        if resolved is None:
            resolved = set()
        start = 'if {cond}:'.format(cond=cond)
        end = ''
        ControlBlock.__init__(self, start, end, content, dependencies, resolved)

class CIfBlock(IfBlock):
    def __init__(self, cond, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set(get_identifiers(cond))
        if resolved is None:
            resolved = set()
        start = 'if({cond})\n{{'.format(cond=cond)
        end = '}'
        ControlBlock.__init__(self, start, end, content, dependencies, resolved)

if __name__=='__main__':
    statements = statements_from_codestring('''
        x := y + z**2
        y = 7*x
        z += 3*exp(y)
        x = 2*(x<y)
        ''')
    block = Block(*statements)
    language = CLanguage()
    #language = PythonLanguage()
    print block.convert_to(language)
    print block.dependencies
    print block.resolved
    print
    if language.name=='python':
        forblock = PythonForBlock('m', 'M[I]', block)
    elif language.name=='c':
        inforblock = Block(CDefineFromArray('m', 'M', 'idx'),
                           block)
        forblock = CIterateArray('idx', 'I', 'I_len', inforblock,
                          dtype='int', reference=False)
    print forblock.convert_to(language)
    print forblock.dependencies
    print forblock.resolved
