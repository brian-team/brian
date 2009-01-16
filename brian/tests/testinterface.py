"""Tests that the interface works as documented

This is a very important test that should be periodically run whenever a significant
change to the underlying code is made. This test serves, along with the documentation,
as a definition of the interface. If the user follows the example of code in this
test, their code shouldn't break when we change details of how Brian works.
"""

import unittest
import new

from brian import *
from brian.globalprefs import get_global_preference
from brian.log import *

class TestSequenceFunctions(unittest.TestCase):
    def set_up(self):
        pass
    
    def testclock(self):
        """Tests the interface to the clock module.
        """        
        from brian.clock import _define_and_test_interface
        _define_and_test_interface(self)
        
    def testmagic(self):
        """Tests the interface to the magic module.
        """
        from brian.magic import _define_and_test_interface
        _define_and_test_interface(self)
        
    def testunits(self):
        """Tests the interface to the units module.
        """        
        from brian.units import _define_and_test_interface
        _define_and_test_interface(self)
        
    def testdirectcontrol(self):
        """Tests the interface to the directcontrol module.
        """
        from brian.directcontrol import _define_and_test_interface
        _define_and_test_interface(self) 
            
    def testunitsafefunctions(self):
        """Tests the interface to the unitsafefunctions module.
        """
        from brian.unitsafefunctions import _define_and_test_interface
        _define_and_test_interface(self)   
        
    def testreset(self):
        """Tests the interface to the reset module.
        """
        from brian.reset import _define_and_test_interface
        _define_and_test_interface(self)  
         
    def testthreshold(self):
        """Tests the interface to the threshold module.
        """
        from brian.threshold import _define_and_test_interface
        _define_and_test_interface(self)
        
    def testmonitor(self):   
        """Tests the interface to the monitor module.
        """
        from brian.monitor import _define_and_test_interface
        _define_and_test_interface(self)
        
    def testconnection(self):   
        """Tests the interface to the connection module.
        """
        from brian.connection import _define_and_test_interface
        _define_and_test_interface(self)
        
    def testinspection(self):
        """Tests the interface to the inspection module.
        """
        from brian.inspection import _define_and_test_interface
        _define_and_test_interface(self)

def run_test():
    log_level_error()
    import inspect, brian, sys
    print '***************************'
    print 'Running the interface tests'
    print '***************************'
    print
    print 'Running from directory:', inspect.getsourcefile(brian)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    return unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite).wasSuccessful()    

if __name__=="__main__":
    run_test()