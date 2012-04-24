import itertools

from brian import *
from nose.tools import *
import numpy

def test():
    """
    Names
    ~~~~~
    
    The following units should exist:
    
        metre, kilogram, second, amp, kelvin, mole, candle
        radian, steradian, hertz, newton, pascal, joule, watt,
        coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
        celsius, lumen, lux, becquerel, gray, sievert, katal,
        gram, gramme
    
    In addition, all versions of these units scaled by the following
    prefixes should exist (in descending order of size):
    
        Y, Z, E, P, T, G, M, k, h, da, d, c, m, u, n, p, f, a, z, y
    
    And, all of the above units with the suffixes 2 and 3 exist, and
    refer to the unit to the power of 2 and 3, e.g. ``metre3 = metre**3``.
    
    Arithmetic
    ~~~~~~~~~~
    
    The following operations on :class:`Quantity` objects require that the operands
    have the same dimensions:
    
        +, -, <, <=, >, >=, ==, !=
    
    The following operations on :class:`Quantity` objects work with any pair of
    operands:
    
        / and *
    
    In addition, ``-x`` and ``abs(x)`` will work on any :class:`Quantity` ``x``, and will return
    values with the same dimension as their argument.
    
    The power operation ``x**y`` requires that ``y`` be dimensionless.
    
    Casting
    ~~~~~~~

    The three rules that define the casting operations for
    :class:`Quantity` object are:
    
    1.  :class:`Quantity` op :class:`Quantity` = :class:`Quantity`:
        Performs dimension consistency check if appropriate.
    2.  Scalar op :class:`Quantity` = :class:`Quantity`: 
        Assumes that the scalar is dimensionless
    3.  other op :class:`Quantity` = other:
        The :class:`Quantity` object is downcast to a ``float``
    
    Scalar types are 1 dimensional number types, including ``float``, ``int``, etc.
    but not ``array``.
    
    The :class:`Quantity` class is a derived class of ``float``, so many other operations
    will also downcast to ``float``. For example, ``sin(x)`` where ``x`` is a quantity
    will return ``sin(float(x))`` without doing any dimension checking. Although
    see the Brian.unitsafefunctions module for a way round this. It is better
    to be explicit about casting if you can be.
    
    TODO: more details on ``numpy.array``/:class:`Quantity` operations?
    
    :func:`check_units` decorator
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    The decorator :func:`check_units` can be used to check that the arguments passed
    to a function have the right units, e.g.::
    
        @check_units(I=amp,V=volt)
        def find_resistance(I,V):
            return V/I
    
    will work if you try ``find_resistance(1*amp,1*volt)`` but raise an exception
    if you try ``find_resistance(1,1)`` say.
    """
    reinit_default_clock()

    # the following units should exist:
    units_which_should_exist = [ metre, meter, kilogram, second, amp, kelvin, mole, candle,
                                radian, steradian, hertz, newton, pascal, joule, watt,
                                coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
                                celsius, lumen, lux, becquerel, gray, sievert, katal,
                                gram, gramme ]
    # scaled versions of all these units should exist (we just check farad as an example)
    some_scaled_units = [ Yfarad, Zfarad, Efarad, Pfarad, Tfarad, Gfarad, Mfarad, kfarad,
                         hfarad, dafarad, dfarad, cfarad, mfarad, ufarad, nfarad, pfarad,
                         ffarad, afarad, zfarad, yfarad ]
    # and check that the above is in descending order of size
    import copy
    sorted_scaled_units = copy.copy(some_scaled_units)
    sorted_scaled_units.sort(reverse=True)
    assert some_scaled_units == sorted_scaled_units
    # some powered units
    powered_units = [ cmetre2, Yfarad3 ]

    # check that operations requiring consistent units work with consistent units
    a = 1 * kilogram
    b = 2 * kilogram
    c = [ a + b, a - b, a < b, a <= b, a > b, a >= b, a == b, a != b ]
    # check that given inconsistent units they raise an exception
    from operator import add, sub, mul, div, lt, le, gt, ge, eq, ne
    tryops = [add, sub, lt, le, gt, ge, eq, ne]
    a = 1 * kilogram
    b = 1 * second
    def inconsistent_operation(a, b, op):
        return op(a, b)
    for op in tryops:
        assert_raises(DimensionMismatchError, inconsistent_operation, a, b, op)

    # check that comparisons to integer zero (no units) work
    a = 1 * kilogram
    assert(all([not (a < 0), not (a <= 0), a > 0, a >= 0, not (a == 0), a != 0]))

    # check that comparisons to float zero (no units) work
    a = 1 * kilogram
    assert(all([not (a < 0.), not (a <= 0.), a > 0., a >= 0., not (a == 0.), a != 0.]))
    
    # check that comparisons to inf and -inf work
    a = 1 * kilogram
    assert(all([a < inf, a <= inf, a > -inf, a >= -inf, a != inf, a != -inf]))
    
    # check that comparisons for dimensionless units to scalars work
    a = 1 * kilogram / kilogram
    assert(all([a < 2, a <= 2, a > 0, a >= 0, a == 1, a != 2]))
    
    # check that operations not requiring consistent units work
    a = 1 * kilogram
    b = 1 * second
    c = [ a * b, a / b ]

    # check that - and abs give results with the same dimensions
    assert (-a).has_same_dimensions(a)
    assert abs(a).has_same_dimensions(a)
    assert - abs(a) < abs(a) # well why not, this should be true

    # check that pow requires the index to be dimensionless
    a = (1 * kilogram) ** 0.352
    def inconsistent_power(a, b):
        return a ** b
    a = 1 * kilogram
    b = 1 * second
    assert_raises(DimensionMismatchError, inconsistent_power, a, b)

    # check casting rule 1
    a = 1 * kilogram
    b = 2 * kilogram
    for op in [add, sub, mul, div]: assert isinstance(op(a, b), Quantity)

    # check casting rule 2
    a = 1
    b = 1 * kilogram
    for op in [mul, div]: assert isinstance(op(a, b), Quantity) and isinstance(op(b, a), Quantity)
    for op in [add, sub]:
        assert_raises(DimensionMismatchError, inconsistent_operation, a, b, op)
    for op in [add, sub, mul, div]: assert isinstance(op(a, b / b), Quantity) and isinstance(op(b / b, a), Quantity)

    # check casting rule 3
    assert isinstance(numpy.array([1, 2]) * (1 * kilogram), numpy.ndarray)
    assert isinstance((1 * kilogram) * numpy.array([1, 2]), numpy.ndarray)

    # check_units decorator
    @check_units(I=amp, V=volt)
    def find_resistance(I, V):
        return V / I
    R = find_resistance(1 * amp, 1 * volt)
    assert_raises(DimensionMismatchError, find_resistance, 1, 1)

    # check that str works (ignoring the result) and that repr returns a 
    # consistent representation, i.e. eval(repr(x)) == x
    
    # Combined units
    complex_units = [(kgram * metre2)/(amp * second3),
                     5 * (kgram * metre2)/(amp * second3),
                     metre * second**-1, 10 * metre * second**-1] 
    for u in itertools.chain(units_which_should_exist, some_scaled_units,
                              powered_units, complex_units):
        str(u)
        assert(eval(repr(u)) == u) 

if __name__ == '__main__':
    test()
