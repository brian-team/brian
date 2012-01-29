def language_method_default(origmethod):
    print origmethod.__name__
    origmethodname = origmethod.__name__
    def meth(self, *args, **kwds):
        methodname = origmethodname+'1'
        if methodname in self.__dict__:
            print 'indict'
            return getattr(self, methodname)(*args, **kwds)
        else:
            print 'notindict', methodname, origmethodname
            #return getattr(self, origmethodname)(*args, **kwds)
            return origmethod(self, *args, **kwds)
    return meth

class A(object):
    @language_method_default
    def f(self):
        return 'A.f'
    def __getattr__(self, name):
        print name
        if name.endswith('1'):
            return getattr(self, name[:-1])
        raise AttributeError(name)

class B(A):
    pass
#    def f(self):
#        return 'B.f'

class C(B):
    def f1(self):
        print 'here'
        return 'C.f1+'+B.f(self)

print C.__mro__
    
print B.__dict__.keys()    

#c = B()
#print c.f1()
