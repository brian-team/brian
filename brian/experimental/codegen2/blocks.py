'''

'''
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
    '''
    Contains a list of :class:`CodeItem` objects which are considered to be
    executed in serial order. The list is passed as arguments to the
    init method, so if you want to pass a list you can initialise as::
    
        block = Block(*items)
    '''
    def __init__(self, *args):
        self.contents = list(args)
    # we only need to implement __iter__ because the default behaviour of
    # CodeItem will concatenate the elements contained within
    def __iter__(self):
        return iter(self.contents)

class ControlBlock(Block):
    '''
    Helper class used as the base for various control structures such as for
    loops, if statements. These are typically not language-invariant and
    should only be output in the resolution process by symbols (which know the
    language they are resolving to). Consists of strings ``start`` and ``end``,
    a list of ``contents`` (as for :class:`Block`), and explicit sets of 
    ``dependencies`` and ``resolved`` (these are self-dependencies/resolved).
    The output code consists of the start string, the indented converted
    contents, and then the end string. For example, for a C for loop, we would
    have ``start='for(...){`` and ``end='}'``.
    '''
    def __init__(self, start, end, contents, dependencies, resolved):
        self.selfdependencies = dependencies
        self.selfresolved = resolved
        self.start = start
        self.end = end
        if isinstance(contents, CodeItem):
            contents = [contents]
        self.contents = contents
    def convert_to(self, language, symbols={}, namespace={}):
        contentcode = Block.convert_to(self, language, symbols=symbols,
                                       namespace=namespace)
        s = self.start+'\n'+indent_string(contentcode)+'\n'+self.end
        return strip_empty_lines(s)

class ForBlock(ControlBlock):
    '''
    Simply a base class, does nothing.
    '''
    pass

class PythonForBlock(ForBlock):
    '''
    A for loop in Python, the structure is::
    
        for var in container:
            content
            
    Where ``var`` and ``container`` are strings, and ``content`` is a
    :class:`CodeItem` or list of items.
    
    Dependencies can be given explicitly, or by default they are ``Read(x)`` for
    each word ``x`` in ``container``. Resolved can be given explicitly, or by
    default it is ``set(var)``.
    '''
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
    '''
    A for loop in C, the structure is::
    
        for(spec)
        {
            content
        }
        
    You specify a string ``var`` which is the variable the loop is iterating
    over, and a string ``spec`` should be of the form ``'int i=0; i<n; i++'``.
    The ``content`` is a :class:`CodeItem` or list of items. The dependencies
    and resolved sets can be given explicitly, or by default they are extracted,
    respectively, from the set of words in ``spec``, and ``set([var])``.
    '''
    def __init__(self, var, spec, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set([Read(x) for x in get_identifiers(spec)])
            dependencies.discard(Read(var))
        if resolved is None:
            resolved = set([var])
        start = 'for({spec})\n{{'.format(spec=spec)
        end = '}'
        ControlBlock.__init__(self, start, end, content, dependencies, resolved)

class IfBlock(ControlBlock):
    '''
    Just a base class.
    '''
    pass

class PythonIfBlock(IfBlock):
    '''
    If statement in Python, structure is::
    
        if cond:
            content
            
    Dependencies can be specified explicitly, or are automatically extracted as
    the words in string ``cond``, and resolved can be specified explicitly or by
    default is ``set()``.
    '''
    def __init__(self, cond, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set(get_identifiers(cond))
        if resolved is None:
            resolved = set()
        start = 'if {cond}:'.format(cond=cond)
        end = ''
        ControlBlock.__init__(self, start, end, content, dependencies, resolved)

class CIfBlock(IfBlock):
    '''
    If statement in C, structure is::
    
        if(cond)
        {
            content
        }
        
    Dependencies can be specified explicitly, or are automatically extracted as
    the words in string ``cond``, and resolved can be specified explicitly or by
    default is ``set()``.
    '''
    def __init__(self, cond, content, dependencies=None, resolved=None):
        if dependencies is None:
            dependencies = set([Read(x) for x in get_identifiers(cond)])
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
        # no longer used
        forblock = CIterateArray('idx', 'I', 'I_len', inforblock,
                          dtype='int', reference=False)
    print forblock.convert_to(language)
    print forblock.dependencies
    print forblock.resolved
