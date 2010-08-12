import copy
from itertools import izip
import itertools
from random import sample
import bisect
from ..units import *
import types
from .. import magic
from ..log import *
from numpy import *
from scipy import sparse, stats, rand, weave, linalg
import scipy
import scipy.sparse
import numpy
from numpy.random import binomial, exponential
import random as pyrandom
from scipy import random as scirandom
from ..utils.approximatecomparisons import is_within_absolute_tolerance
from ..globalprefs import *
from ..base import *
from ..stdunits import ms
from operator import isSequenceType

effective_zero = 1e-40

colon_slice = slice(None, None, None)

def todense(x):
    if hasattr(x, 'todense'):
        return x.todense()
    return array(x, copy=False)
