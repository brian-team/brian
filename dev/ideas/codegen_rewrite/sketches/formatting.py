from string import Formatter

class MyFormatter(Formatter):
    def __init__(self, namespace=None):
        Formatter.__init__(self)
        if namespace is None:
            namespace = {}
        self.namespace = namespace
    def get_value(self, first, args, kwargs):
        try:
            return Formatter.get_value(self, first, args, kwargs)
        except KeyError:
            return eval(first, self.namespace)

s = '''
for(int i=0; i<N; i++)
{{
    cout << i << endl;
}}
{f(3*5)}
'''

def f(x):
    return x*x

print MyFormatter(globals()).format(s)

#print '{x_{y}}'.format(y='z', x_z='7')