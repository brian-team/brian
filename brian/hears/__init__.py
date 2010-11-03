import brian as _brian
if _brian.get_global_preference('usenewbrianhears'):
    import warnings as _warnings
    _warnings.warn('Using new version of Brian hears')
    from .newversion import *
else:
    from .sounds import *
    from .erb import *
    from .filtering import *
    from .hrtf import *

    dB = 1.
