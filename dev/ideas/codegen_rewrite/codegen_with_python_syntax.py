import re, inspect

# We just use this class to identify certain variables
class Symbol(str): pass

# This class is basically just a mutable string
class OutputCode(object):
    def __init__(self):
        self.code=''

    def __iadd__(self, code):
        self.code=self.code+code
        return self

# Iterating over instances of this class generates code
# for iterating over a C++ array, it yields a single
# Symbol object, the variable name of the value in the
# array
class Array(object):
    def __init__(self, code, name, dtype='double'):
        self.name=name
        self.dtype=dtype
        self.code=code

    def __iter__(self):
        def f():
            self.code+='for(int {name}_index=0; {name}_index<{name}_len; {name}_index++){{\n'.format(name=self.name)
            self.code+='{dtype} &{name} = {name}_array[{name}_index];\n'.format(dtype=self.dtype, name=self.name)
            yield Symbol(self.name)
            self.code+='}\n'
        return f()

# Instances of this class generate C++ code from Python syntax
# code snippets, replacing variable names that are a Symbol in the
# namespace with the value of that Symbol.
class Evaluator(object):
    def __init__(self, code):
        self.code=code

    def __call__(self, code):
        # The set of variables in the code snippet
        vars=re.findall(r'\b(\w+)\b', code)
        # Extract any names from the namespace of the calling frame
        frame=inspect.stack()[1][0]
        globals, locals=frame.f_globals, frame.f_locals
        values={}
        for var in vars:
            if var in locals:
                values[var]=locals[var]
            elif var in globals:
                values[var]=globals[var]
        # Replace any variables whose values are Symbols with their values
        for var, value in values.iteritems():
            if isinstance(value, Symbol):
                code=re.sub(r'\b{var}\b'.format(var=var), str(value), code)
        # Turn Python snippets into C++ (just a simplified version for now)
        code=code.replace(';', '\n')
        lines=[line.strip() for line in code.split('\n')]
        code=''.join(line+';\n' for line in lines)
        self.code+=code

if __name__=='__main__':
    code=OutputCode()
    evaluate=Evaluator(code)
    X=Array(code, 'values')
    for x in X:
        evaluate('x += 5; x *= 2')
    print code.code
