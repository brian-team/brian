from ..log import *
import atexit

__all__ = ['set_gpu_device', 'close_cuda']
    
try:
    import pycuda
    import pycuda.driver as drv
    pycuda.context = None
    drv.init()
    MAXGPU = drv.Device.count()
    
    def set_gpu_device(n):
        """
        This function makes pycuda use GPU number n in the system.
        """
        log_debug('brian.hears', "Setting PyCUDA context number %d" % n)
        try:
            pycuda.context.detach()
        except:
            pass
        pycuda.context = drv.Device(n).make_context()
    
    def close_cuda():
        """
        Closes the current context. MUST be called at the end of the script.
        """
        if pycuda.context is not None:
            log_debug('brian.hears', "Closing current PyCUDA context")
            try:
                pycuda.context.pop()
                pycuda.context = None
            except:
                pass
    
    atexit.register(close_cuda)

except:
    MAXGPU = 0
    
    def set_gpu_device(n):
        raise Exception("PyCUDA not available")
        pass

    def close_cuda():
        pass

#if __name__ == '__main__':
#    close_cuda()