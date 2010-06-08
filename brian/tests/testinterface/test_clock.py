from brian import *
from nose.tools import *

def test():
    """
    The :class:`Clock` object
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    
    A :class:`Clock` object is initialised via::
    
        c = Clock(dt=0.1*msecond, t=0*msecond, makedefaultclock=False)
    
    In particular, the following will work and do the same thing::
    
        c = Clock()
        c = Clock(t=0*second)
        c = Clock(dt=0.1*msecond)
        c = Clock(0.1*msecond,0*second)
    
    Setting the ``makedefaultclock=True`` argument sets the newly
    created clock as the default one.

    The default clock
    ~~~~~~~~~~~~~~~~~
    
    The default clock can be found using the :func:`get_default_clock` function,
    and redefined using the :func:`define_default_clock` function, where the
    arguments passed to :func:`define_default_clock` are the same as the
    initialising arguments to the ``Clock(...)`` statement. The default
    clock can be reinitialised by calling :func:`reinit_default_clock`.
    
    A less safe way to access the default clock is to refer directly to
    the variable :data:`defaultclock`. If the default clock has been redefined, this
    won't work.

    The :func:`guess_clock` function
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    The function :func:`guess_clock` should return (in this order of priority):
    
    * The clock passed as an argument, if one was passed
    * A clock defined in the calling function, if there was one
    * The default clock otherwise
    
    If more than one clock was defined in the calling function, it should
    raise a ``TypeError`` exception.
    """
    reinit_default_clock()

    # check that 'defaultclock' default clock exists and starts at t=0
    assert defaultclock.t < 0.001 * msecond

    # check that default clock exists and starts at t = 0
    c = guess_clock()
    assert c.t < 0.001 * msecond

    # check that passing no arguments works
    c = Clock()

    # check that passing t argument works
    c = Clock(t=1 * second)
    assert c.t > 0.9 * second

    # check that passing dt argument works
    c = Clock(dt=10 * msecond)
    assert c.dt > 9 * msecond

    # check that passing t and dt arguments works
    c = Clock(t=2 * second, dt=1 * msecond)
    assert c.t > 1.9 * second
    assert 0.5 * msecond < c.dt < 2 * msecond

    # check that making this the default clock works
    assert get_global_preference('defaultclock').dt < 9 * msecond
    c = Clock(dt=10 * msecond, makedefaultclock=True)
    assert get_global_preference('defaultclock').dt > 9 * msecond

    # check that the other ways of defining a default clock work
    define_default_clock(dt=3 * msecond)
    assert 2.9 * msecond < get_global_preference('defaultclock').dt < 3.1 * msecond

    # check that the get_default_clock function works
    assert 2.9 * msecond < get_default_clock().dt < 3.1 * msecond

    # check that passing unnamed arguments in the order dt, t works
    c = Clock(10 * msecond, 1 * second)
    assert c.t > 0.9 * second
    assert c.dt > 9 * msecond

    # check that reinit() sets t=0
    c = Clock(t=1 * second)
    c.reinit()
    assert c.t < 0.001 * second

    # check that tick() works
    c = Clock(t=1 * second, dt=1 * msecond)
    for i in range(10): c.tick()
    assert c.t > 9 * msecond

    # check that reinit_default_clock works
    for i in range(10): get_default_clock().tick()
    reinit_default_clock()
    assert get_default_clock().t < 0.0001 * second

    # check that guess_clock passed a clock returns that clock
    assert 0.6 * second < guess_clock(Clock(t=0.7 * second)).t < 0.8 * second

    # check that guess_clock passed no clock returns the only clock we've defined in this function so far
    c = Clock()
    assert guess_clock() is c

    # check that if no clock is defined, guess_clock returns the default clock
    del c
    assert guess_clock() is get_default_clock()

    # check that if two or more clocks are defined, guess_clock raises a TypeError
    c = Clock()
    d = Clock()
    assert_raises(TypeError, guess_clock)
    del d

    # check that if we have a calling stack, the innermost clock is found
    c = Clock()
    def f():
        d = Clock()
        assert guess_clock() is d
        del d
        return guess_clock()
    assert f() is c

    # cleanup: reset the default clock to its default state
    set_global_preferences(defaultclock=defaultclock)
