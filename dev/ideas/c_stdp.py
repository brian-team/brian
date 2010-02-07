'''
Notes:

The main problem for STDP does indeed appear to be the running of the STDP code
and not the get_past_values_seq method of RecentStateMonitor. So, code generation
and optimisation efforts should be focussed on the code in stdp.py.
'''

from brian import *
from scipy import weave

