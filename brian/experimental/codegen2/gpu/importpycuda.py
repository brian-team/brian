try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as driver
    import pycuda.compiler as compiler
    from pycuda import gpuarray
    from pycuda import scan
except ImportError:
    pycuda = None
