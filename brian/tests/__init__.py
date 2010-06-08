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

if __name__ == '__main__':
    go()
