'''
Runs all of the examples in the examples/ folder, except those excluded in 
examples_excluded.txt or those run previously (as recorded in
examples_completed.txt).

Note: Uses the 'Agg' backend, therefore no plots are displayed on the screen and
it is possible to run this script on a headless server.
'''

import os, nose, warnings
from nose.plugins import Plugin
from nose.plugins.capture import Capture
from nose.plugins.logcapture import LogCapture
from nose.plugins.cover import Coverage
from nose.plugins.xunit import Xunit

# Use the 'Agg' backend, normally used for saving plots to files
# this has the advantage of no plots showing up and can be run 
# without an X server or similar. This has to be done before importing brian
# because brian imports pylab
import matplotlib as _mpl
_mpl.use('Agg')

# Suppress all warnings
warnings.simplefilter('ignore')
import brian # It seems if we import brian here, nose can capture the logging

exclude_list = open('examples_exclude.txt', 'U').read().split('\n')
exclude_list = [f for f in exclude_list if not f.startswith('#') and f]

if os.path.exists('examples_completed.txt'):
    examples_completed = [f for f in open('examples_completed.txt', 'U').read().split('\n') if f]
else:
    examples_completed = []

class SelectFilesPlugin(Plugin):
    '''
    This plugin makes nose descend into all directories but skips files that
    are mentioned in examples_exclude.txt or examples_completed.txt
    '''
    # no command line arg needed to activate plugin
    enabled = True
    name = "select-files"

    def configure(self, options, conf):
        pass # always on

    def wantDirectory(self, dirname):
        # we want all directories
        return True
    
    def wantFile(self, filename):
        # we only need the filename, not the path
        filename = os.path.split(filename)[1]
        if filename in exclude_list or filename in examples_completed:
            return False
        elif filename.endswith('.py'):
            return True
        else:
            return False
        
class RecordCompletionPlugin(Plugin):
    ''' 
    This plugin saves the completed examples to the file examples_completed.txt,
    already completed examples will be skipped on subsequent runs.
    
    Note that this is ignored by the automatic testing on the Jenkins server, as
    the script in dev/jenkins/linux_scripts/run_examples.sh first deletes
    examples_completed.txt
    '''
    
    enabled = True
    name = 'record-completion'

    def configure(self, options, conf):
        pass # always on

    def afterImport(self, filename, module):
        filename = os.path.split(filename)[1]
        open('examples_completed.txt', 'a').write(filename + '\n')
        
argv = [__file__, '-v', '--logging-clear-handlers', '--with-xunit',
        '--with-coverage', '--cover-html', '--cover-package=brian',
        '../../examples']

nose.main(argv=argv, plugins=[SelectFilesPlugin(), RecordCompletionPlugin(),
                              LogCapture(), Capture(), Coverage(), Xunit()])