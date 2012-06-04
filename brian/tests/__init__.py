from brian import *
import brian
import os

def go():
    try:
        import nose
    except ImportError:
        print "Brian testing framework uses the 'nose' package."
    print 'Brian running from file:', brian.__file__
    # For running tests from an IPython shell, use magic_useframes=True, but we
    # restore the state after running 
    magic_useframes = get_global_preference('magic_useframes')
    set_global_preferences(magic_useframes=True)
    nose.config.logging.disable(nose.config.logging.ERROR)
    basedir, _ = os.path.split(__file__)
    cwd = os.getcwd()
    os.chdir(basedir)
    nose.run()
    os.chdir(cwd)
    set_global_preferences(magic_useframes=magic_useframes)

def repeat_with_global_opts(opt_list):
    '''
    This decorator is used for testing with different sets of global options.
    Decorate a test function (note that the test function should not have any
    return value; when using this decorator, nothing is returned) with it and
    give a list of dictionaries as parameters, where each dictionary consists
    of keyword/value combinations for the `set_global_preferences` function.
    The global preferences are reset to their previous values after the test
    run.
    
    Example usage:
        
        @repeat_with_global_opts([{'useweave': False}, {'useweave': True}])
        def test_something():
            ...
    '''
    def decorator(func):
        def wrapper(*args, **kwds):
            for opts in opt_list:
                # save old preferences
                old_preferences = get_global_preferences()
                set_global_preferences(**opts)
                print 'Repeating test %s with options: %s' % (func.__name__, opts)
                func(*args, **kwds)
                # reset preferences
                set_global_preferences(**old_preferences)

        #make sure that the wrapper has the same name as the original function
        #otherwise nose will ignore the functions as they are not called
        #"test..." anymore!
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        
        return wrapper
    
    return decorator
        
if __name__ == '__main__':
    go()
