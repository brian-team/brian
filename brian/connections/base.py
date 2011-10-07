import copy
from itertools import izip
import itertools
from random import sample
import bisect
from ..units import second, msecond, check_units, DimensionMismatchError
import types
from .. import magic
from ..log import log_warn, log_info, log_debug
from numpy import *
from scipy import sparse, stats, rand, weave, linalg
import scipy
import scipy.sparse
import numpy
from numpy.random import binomial, exponential
import random as pyrandom
from scipy import random as scirandom
from ..utils.approximatecomparisons import is_within_absolute_tolerance
from ..globalprefs import get_global_preference
from ..base import ObjectContainer
from ..stdunits import ms
from operator import isSequenceType

effective_zero = 1e-40

colon_slice = slice(None, None, None)

def todense(x):
    if hasattr(x, 'todense'):
        return x.todense()
    return array(x, copy=False)
