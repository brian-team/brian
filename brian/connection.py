'''
This module is here only to maintain backward compatibility with old code that
references brian.connection rather than brian.connections. It should be removed
at some point in the future
'''

from connections import *

import log as _log
_log.log_warn('brian', 'Module "connection" in Brian is deprecated, use "brian.connections" instead.')
