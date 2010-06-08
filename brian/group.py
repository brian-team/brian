from brian import *

__all__ = ['Group', 'MultiGroup']


class Group(object):
    '''
    Generic fixed-length variable values container class
    
    Used internally by Brian to store values associated with variables. Primary
    use is for NeuronGroup to store a state matrix of values associated to the
    different variables.
    
    Each differential equation in ``equations`` is allocated an array of
    length ``N`` in the attribute ``_S`` of the object. Unit consistency
    checking is performed.
    '''
    def __init__(self, equations, N, level=0, unit_checking=True):
        if isinstance(equations, str):
            equations = Equations(equations, level=level + 1)
        equations.prepare(check_units=unit_checking)
        var_names = equations._diffeq_names
        M = len(var_names)
        self._S = zeros((M, N))
        self.staticvars = dict([(name, equations._function[name]) for name in equations._eq_names])
        self.var_index = dict(zip(var_names, range(M)))
        self.var_index.update(zip(range(M), range(M))) # name integer i -> state variable i
        for var1, var2 in equations._alias.iteritems():
            self.var_index[var1] = self.var_index[var2]

    def get_var_index(self, name):
        '''
        Returns the index of state variable "name".
        '''
        return self.var_index[name]

    def __len__(self):
        '''
        Number of neurons in the group.
        '''
        return self._S.shape[1]

    def num_states(self):
        return self._S.shape[0]

    def state_(self, name):
        '''
        Gets the state variable named "name" as a reference to the underlying array
        '''
        if isinstance(name, int):
            return self._S[name]
        if name in self.staticvars:
            f = self.staticvars[name]
            return f(*[self.state_(var) for var in f.func_code.co_varnames])
        i = self.var_index[name]
        return self._S[i]
    state = state_

    def __getattr__(self, name):
        if name == 'var_index':
            # this seems mad - the reason is that getattr is only called if the thing hasn't
            # been found using the standard methods of finding attributes, which for var_index
            # should have worked, this is important because the next line looks for var_index
            # and if we haven't got a var_index we don't want to get stuck in an infinite
            # loop
            raise AttributeError
        if not hasattr(self, 'var_index'):
            # only provide lookup of variable names if we have some variable names, i.e.
            # if the var_index attribute exists
            raise AttributeError
        try:
            return self.state(name)
        except KeyError:
            if len(name) and name[-1] == '_':
                try:
                    origname = name[:-1]
                    return self.state_(origname)
                except KeyError:
                    raise AttributeError
            raise AttributeError

    def __setattr__(self, name, val):
        origname = name
        if len(name) and name[-1] == '_':
            origname = name[:-1]
        if not hasattr(self, 'var_index') or (name not in self.var_index and origname not in self.var_index):
            object.__setattr__(self, name, val)
        else:
            if name in self.var_index:
                self.state(name)[:] = val
            else:
                self.state_(origname)[:] = val


class MultiGroup(object):
    '''
    Generic variable-length variable values container class
    
    Used internally by Brian to store values associated with variables.
    See :class:`Group` for more details.
    
    This class takes a list of groups or of ``(equations, N)`` pairs.
    The ``state()`` and ``state_()`` methods work as if it were a
    single :class:`Group` but return arrays of different lengths from
    different state matrix structures depending on where the state
    value is defined. Similarly for attribute access.
    '''
    def __init__(self, groups, level=0):
        newgroups = []
        self.var_index = dict()
        for g in groups:
            if not isinstance(g, Group):
                eqs, N = g
                g = Group(eqs, N, level=level + 1)
            newgroups.append(g)
            self.var_index.update(dict((k, g) for k in g.var_index.keys()))
            self.var_index.update(dict((k, g) for k in g.staticvars.keys()))
        self.groups = newgroups

    def state_(self, name):
        return self.var_index[name].state_(name)
    state = state_

    def __getattr__(self, name):
        try:
            return self.state(name)
        except KeyError:
            if len(name) and name[-1] == '_':
                try:
                    origname = name[:-1]
                    return self.state_(origname)
                except KeyError:
                    raise AttributeError
            raise AttributeError

    def __setattr__(self, name, val):
        origname = name
        if len(name) and name[-1] == '_':
            origname = name[:-1]
        if not hasattr(self, 'var_index') or (name not in self.var_index and origname not in self.var_index):
            object.__setattr__(self, name, val)
        else:
            if name in self.var_index:
                self.state(name)[:] = val
            else:
                self.state_(origname)[:] = val

if __name__ == '__main__':
    if True:
        eqs_A = Equations('''
            dV/dt = -V/(1*second) : volt
            W = V*V : volt2
            x = V
            ''')
        eqs_B = Equations('''
            dVb/dt = -Vb/(1*second) : volt
            Wb = Vb*Vb : volt2
            xb = Vb
            ''')
        g = MultiGroup(((eqs_A, 10), (eqs_B, 5)))
        for h in g.groups:
            h._S[:] = rand(*h._S.shape)
        print g.V
        print g.V ** 2
        print g.W
        print g.x
        print g.Vb
        print g.Vb ** 2
        print g.Wb
        print g.xb
    if False:
        eqs = Equations('''
            dV/dt = -V/(1*second) : volt
            W = V*V : volt2
            x = V
            ''')
        g = Group(eqs, 10)
        print g._S
        g._S[:] = rand(*g._S.shape)
        print g._S
        print g.V
        print g.V ** 2
        print g.W
        print g.x
        g.V = -1
        print g.V
        print g.W
        print g.x
