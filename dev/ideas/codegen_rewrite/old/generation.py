from brian import *
from expressions import *
from formatter import *
from codeobject import *
from brian.utils.documentation import flattened_docstring, indent_string

__all__ = ['CodeBlock',
           ]

class CodeBlock(object):
    def __init__(self, resolved, content, tabs=0):
        self.resolved = resolved
        if isinstance(content, str) and '\n' in content:
            content = [flattened_docstring(content)]
        self.content = content
        self.tabs = tabs
        
    def __str__(self):
        s = ''
        for c in self.content:
            if isinstance(c, Statement):
                c = indent_string(str(c), self.tabs)
            elif isinstance(c, str):
                c = indent_string(c, self.tabs)
            else:
                c = str(c)+'\n'
            s = s+c
        return s
    
    def indented(self, tabs=1):
        return CodeBlock(self.resolved, self.content, self.tabs+tabs)
    
    def generate_codestr(self, language, symbols, symbols_to_load=None):
        codestr = ''
        # first we generate code to load whichever variables
        # can be resolved (all dependencies met)
        if symbols_to_load is None:
            symbols_to_load = symbols
        new_symbols_to_load = {}
        for name, sym in symbols_to_load.items():
            deps = sym.depends
            if set(deps).issubset(set(self.resolved)):
                codestr += indent_string(sym.load, self.tabs)
            else:
                new_symbols_to_load[name] = sym
        # now we generate the content (recursively)
        for item in self.content:
            itemstr = None
            if isinstance(item, str):
                itemstr = indent_string(item, self.tabs)
            if isinstance(item, Statement):
                itemstr = item.convert_to(language, symbols)
                itemstr = indent_string(itemstr, self.tabs)
            if isinstance(item, CodeBlock):
                itemstr = item.generate_codestr(language, symbols,
                                                new_symbols_to_load)
            if itemstr is None:
                raise TypeError("Unknown code block item type")
            codestr = codestr+itemstr
        return codestr
    
    def generate(self, language, symbols):
        codestr = self.generate_codestr(language, symbols)
        namespace = {}
        for name, sym in symbols.iteritems():
            sym.update_namespace(namespace)
        if language.name=='python':
            code = PythonCode(codestr, namespace)
        elif language.name=='c':
            code = CCode(codestr, namespace)
        return code
