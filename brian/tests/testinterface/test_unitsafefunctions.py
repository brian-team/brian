from brian import *
from nose.tools import *
import numpy

def test():
    """
    Each of the following functions f(x) should use units if they are passed a
    :class:`Quantity` object or fall back on their numpy versions
    otherwise.
    
        sqrt, log, exp, sin, cos, tan, arcsin, arccos, arctan,
        sinh, cosh, tanh, arcsinh, arccosh, arctanh    
    """
    reinit_default_clock()
    # check sqrt behaves as expected
    x = 3 * second
    z = numpy.array([3., 2.])
    assert (have_same_dimensions(sqrt(x), second ** 0.5))
    assert (isinstance(sqrt(z), numpy.ndarray))

    # check the return types are right for all other functions
    x = 0.5 * second / second
    funcs = [
        sqrt, log, exp, sin, cos, tan, arcsin, arccos, arctan,
        sinh, cosh, tanh, arcsinh, arccosh, arctanh
            ]
    for f in funcs:
        assert (isinstance(f(x), Quantity))
        assert (isinstance(f(z), numpy.ndarray))

    # check that attempting to use these functions on something with units fails
    funcs = [
        log, exp, sin, cos, tan, arcsin, arccos, arctan,
        sinh, cosh, tanh, arcsinh, arccosh, arctanh
            ]
    x = 3 * second
    for f in funcs:
        assert_raises(DimensionMismatchError, f, x)

if __name__ == '__main__':
    test()
