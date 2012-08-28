import itertools

import numpy as np

from units import *
from stdunits import *

# Construct quantities

# Slicing and indexing, setting items

# Binary operations

# Binary comparisons

# Functions that should not change units
values = [3, np.array([1, 2]), np.ones((3, 3))]
units = [volt, second, siemens, mV, kHz]

keep_dim_funcs = [np.abs, np.cumsum, np.max, np.mean, np.min, np.negative,
                  np.ptp, np.round, np.squeeze, np.std, np.sum, np.transpose]

for value, unit in itertools.product(values, units):
    q_ar = value * unit
    for func in keep_dim_funcs:
        test_ar = func(q_ar)
        if not get_dimensions(test_ar) is q_ar.dim:
            raise AssertionError('%s failed on %s -- dim was %s, is now %s' %
                                 (func.__name__, repr(q_ar), q_ar.dim,
                                  get_dimensions(test_ar)))

# Functions that should change units in a simple way