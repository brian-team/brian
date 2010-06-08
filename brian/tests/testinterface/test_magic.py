from brian import *
from nose.tools import *

def test():
    """
    See the main documentation or the API documentation for details of the
    purpose, main functions and classes of the magic module.
    
    Functions
    ~~~~~~~~~
    
    * :func:`get_instances(instancetype,level=1)`
    * :func:`find_instances(instancetype,startlevel=1)`
    * :func:`find_all_instances(instancetype,startlevel=1)`
    
    Here instancetype is a class derived from :class:`InstanceTracker`, including
    :class:`Clock`, :class:`NeuronGroup`, :class:`Connection`, :class:`NetworkOperation`.
    ``level`` is an integer
    greater than 0 that tells the function how far back it should search
    in the sequence of called functions. ``level=0`` means it should find
    instances from the function calling one of the magic functions,
    ``level=1`` means it should find instances from the function calling the
    function calling one of the magic functions, etc.
    
    :func:`get_instances` returns all instances at a specified level.
    :func:`find_instances`
    searches increasing levels starting from the given ``startlevel`` until it
    finds a nonzero number of instances, and then returns those.
    :func:`find_all_instances` finds all instances from a given level onwards.
    
    Classes
    ~~~~~~~
    
    An object of a class ``cls`` derived from ``InstanceTracker`` will be tracked
    if ``cls._track_instances()`` returns ``True``. The default behaviour of
    ``InstanceTracker`` is to always return ``True``, but this method can be
    redefined if you do not want to track instances of a particular
    subclass of a tracked class. Note that this method is a static method
    of a class, and cannot be used to stop particular instances being
    tracked. Redefine it using something like::
        
        class A(InstanceTracker):
            pass
        class B(A):
            @staticmethod
            def _track_instances(): return False

    A warning (technical detail)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    The current implementation of instance tracking will return
    every variable that was created in a given execution frame (i.e. in
    a given function) that is still alive (i.e. a reference still exists
    to it somewhere). So for example, the following will not find any
    instance::
    
        class A(InstanceTracker):
            pass
        def f(A):
            return A()
        a = f(A)
        print get_instances(A,level=0)
    
    The reason is that the object a was created in the function ``f``, but
    the :func:`get_instances` statement is run in the main body. Similarly, the
    following will return two instances rather than one as you might
    expect::
    
        class A(InstanceTracker):
            pass
        def f(A):
            a = A()
            return get_instances(A,level=0)
        insts1 = f(A)
        isnts2 = f(A)
        print insts2
    
    The reason is that the first call to ``f`` defines an instance of ``A`` and
    returns (via :func:`get_instances`) an object which contains a reference to it.
    The object is still therefore alive. The second call to ``f`` creates a new
    ``A`` object, but the :func:`get_instances` call returns both ``A`` objects, because they
    are both still alive (the first is stored in ``insts1``) and both created
    in the function ``f``.
    
    This behaviour is not part of the assured interface of the magic module,
    and you shouldn't rely on it.
    """
    reinit_default_clock()

    # Define a heirarchy of classes A, B, C, D, E and track instances of them
    # Do not track instances of D or E
    class A(InstanceTracker):
        gval = 0
        def __init__(self):
            self.value = A.gval
            A.gval += 1
        def __repr__(self):
            return str(self)
        def __str__(self):
            return str(self.value)
    class B(A):
        pass
    class C(B):
        pass
    class D(C):
        @staticmethod
        def _track_instances(): return False
    class E(D):
        pass

    # Create some sample objects of each type
    a1 = A() # object 0
    a2 = A() # object 1
    b = B()  # object 2
    c = C()  # object 3
    d = D()  # object 4
    e = E()  # object 5

    # Find the instances of each type on this level
    instA, names = get_instances(A, level=0)
    instB, names = get_instances(B, level=0)
    instC, names = get_instances(C, level=0)
    instD, names = get_instances(D, level=0)
    instE, names = get_instances(E, level=0)

    # This is the expected behaviour:
    # instA = [a1, a2, b, c]
    assert all(o in instA for o in [a1, a2, b, c])
    assert all(o not in instA for o in [d, e])
    # instB = [b, c]
    assert all(o in instB for o in [b, c])
    assert all(o not in instB for o in [a1, a2, d, e])
    # instC = [c]
    assert c in instC
    assert all(o not in instC for o in [a1, a2, b, d, e])
    # instD = instE = []
    assert len(instD) == 0
    assert len(instE) == 0

    # Check that level=0 and level=1 work as expected
    def f1(vars, A, B, C, D, E):
        a3 = A() # object 6
        inst_ahere, names = get_instances(A, level=0) # level=0 should refer to definitions inside f
        inst_abefore, names = get_instances(A, level=1) # level=1 should refer to definitions inside the function calling f
        # inst_abefore = [a1, a2, b, c]
        assert all(o in instA for o in vars[0:4])
        assert all(o not in instA for o in vars[4:])
        # inst_ahere = [a3]
        assert len(inst_ahere) == 1 and a3 in inst_ahere
    f1([a1, a2, b, c, d, e], A, B, C, D, E)

    # Check that nested function calling works as expected
    def f2(A):
        a4 = A() # object 7
        return [get_instances(A, level) for level in range(2)]
    inst = f2(A)
    # inst[0][0] = [a4]
    assert str(inst[0][0]) == '[7]'
    # inst[1][0] = [a1,a2,c,b]
    assert len(inst[1][0]) == 4 and all(o in inst[1][0] for o in [a1, a2, b, c])
    def f3(A):
        a5 = A() # object 9
        return [get_instances(A, level) for level in range(2)]
    def f4(A):
        a6 = A() # object 8
        return f3(A)
    inst = f4(A)
    # inst[0][0] = [a5]
    assert str(inst[0][0]) == '[9]'
    # inst[1][0] = [a6]
    assert str(inst[1][0]) == '[8]'

    # check that find_instances works as expected
    def f5(A):
        return f6(A)
    def f6(A):
        a = A() # object 10
        return f7(A)
    def f7(A):
        return find_instances(A, startlevel=0)
    inst = f5(A)[0][0]
    assert str(inst) == '10'

    # check that find_all_instances works as expected
    def f8(A):
        a = A() # object 11
        return find_all_instances(A, startlevel=0)
    insts = f8(A)[0] # should be objects 0,1,2,3 and 11
    s = map(str, insts)
    s.sort()
    assert str(s) == "['0', '1', '11', '2', '3']"

if __name__ == '__main__':
    test()
