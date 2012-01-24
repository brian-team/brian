from brian import *

class CodeItem(object):
    # Some default values to simplify coding, a class deriving from this can
    # either define selfdependencies/selfresolved or dependencies/resolved.
    # If they define only the self* ones, they also need to define an
    # iterator of contained code items.
    @property
    def subdependencies(self):
        try:
            deps = set()
            for item in self:
                deps.update(item.dependencies)
            return deps
        except Exception, e:
            print e
            raise
    @property
    def subresolved(self):
        try:
            res = set()
            for item in self:
                res.update(item.resolved)
            return res
        except Exception, e:
            print e
            raise
    def __getattr__(self, name):
        if name=='resolved':
            return self.selfresolved.union(self.subresolved)
        elif name=='dependencies':
            return self.selfdependencies.union(self.subdependencies)
        elif name=='selfdependencies':
            return set()
        elif name=='selfresolved':
            return set()
        raise AttributeError(name)
    def __iter__(self):
        return NotImplemented
    def convert_to(self, language, symbols={}):
        return '\n'.join(item.convert_to(language,
                                         symbols=symbols) for item in self)
