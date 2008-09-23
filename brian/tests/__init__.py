import brian

# Tests to run:
import testinterface
import testverification

__all__ = [ 'run_all_tests' ]

def run_all_tests():
    '''
    Run all of Brian's test functions
    '''
    # For running tests from an IPython shell, use magic_useframes=True, but we restore the state
    # after running 
    magic_useframes = brian.get_global_preference('magic_useframes')
    brian.set_global_preferences(magic_useframes=True)
    testset = [v for v in globals().itervalues() if hasattr(v,'run_test')]
    result = True
    print '======================='
    print 'Running all known tests'
    print '======================='
    print
    brian.reinit_default_clock()
    for v in testset:
        result = result & v.run_test()
        brian.reinit_default_clock()
    print
    if result:
        print '======================'
        print 'All tests completed OK'
        print '======================'
    else:
        print '=============================='
        print 'Test failure, see output above'
        print '=============================='
    brian.set_global_preferences(magic_useframes=magic_useframes)
    return result

if __name__=='__main__':
    run_all_tests()
