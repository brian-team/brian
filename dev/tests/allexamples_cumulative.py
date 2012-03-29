'''
Runs all of the examples in the examples/ folder, except those excluded in 
examples_excluded.txt or those run previously (as recorded in
examples_completed.txt).

Note: Uses the 'Agg' backend, therefore no plots are displayed on the screen and
it is possible to run this script on a headless server.
'''

import os, nose, sys, subprocess, warnings, unittest
import tempfile, pickle
from nose.plugins import Plugin
from nose.plugins.capture import Capture
from nose.plugins.logcapture import LogCapture
from nose.plugins.cover import Coverage
from nose.plugins.xunit import Xunit
import warnings
warnings.simplefilter('ignore')
import brian # It seems if we import brian here, nose can capture the logging

exclude_list = open('examples_exclude.txt', 'U').read().split('\n')
exclude_list = [f for f in exclude_list if not f.startswith('#') and f]

if os.path.exists('examples_completed.txt'):
    examples_completed = [f for f in open('examples_completed.txt', 'U').read().split('\n') if f]
else:
    examples_completed = []

class RunTestCase(unittest.TestCase):
    '''
    A test case that simply executes a python script and notes the execution of
    the script in a file `examples_completed.txt`.
    '''
    def __init__(self, filename):
        unittest.TestCase.__init__(self)
        self.filename = filename
    
    def runTest(self):
        # a simpler version of what the nosepipe plugin achieves:
        # isolate test execution in a subprocess:
        tempfilename = tempfile.mktemp('exception')
        
        # Catch any exception and save it to a temporary file
        code_string = """
import matplotlib as _mpl
_mpl.use('Agg')
import warnings
warnings.simplefilter('ignore')
import pickle
try:
    execfile('%s')
except Exception, ex:
    print 'an Exception occured:', ex
    f = open('%s', 'w')
    pickle.dump(ex, f, -1)
    f.close()
""" % (self.filename, tempfilename)
        
        args = [sys.executable, '-c',
                code_string]
        subprocess.call(args)
        
        # Re-raise any exception that occured
        if os.path.exists(tempfilename):
            f = open(tempfilename, 'r')
            ex = pickle.load(f)
            self.successful = False
            raise ex
        else:
            self.successful = True
    
    def tearDown(self):
        if self.successful:
            open('examples_completed.txt', 'a').write(self.filename + '\n')

    def __str__(self):
        return 'Test: ' + self.filename

class SelectFilesPlugin(Plugin):
    '''
    This plugin makes nose descend into all directories but skips files that
    are mentioned in examples_exclude.txt or examples_completed.txt
    Test cases simply consist of executing the script.
    '''
    # no command line arg needed to activate plugin
    enabled = True
    name = "select-files"

    def configure(self, options, conf):
        pass # always on

    def wantDirectory(self, dirname):
        # we want all directories
        return True

    def find_examples(self, name):
        short_name = os.path.split(name)[1]
        examples = []
        if short_name in exclude_list or name in examples_completed:
            return []
        elif os.path.isdir(name):
            for subname in os.listdir(name):
                examples.extend(self.find_examples(os.path.join(name, subname))) 
            return examples
        elif name.endswith('.py'):  # only execute Python scripts
            return [name]
        else:
            return []
        
    def loadTestsFromName(self, name, module=None, discovered=False):        
        all_examples = self.find_examples(name)        
        return [RunTestCase(example) for example in all_examples]
        
        
argv = [__file__, '-v', '--logging-clear-handlers', '--with-xunit',
        '--with-coverage', '--cover-html', '--cover-package=brian',
        '../../examples']

nose.main(argv=argv, plugins=[SelectFilesPlugin(), LogCapture(), Capture(),
                              Coverage(), Xunit()])
