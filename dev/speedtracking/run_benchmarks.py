'''
Run benchmarks for all revisions of Brian in the repository, starting with
version 1.0.0 and only using the last revision of a date, if more than one
commit happened at that day. 
To run the benchmark on your machine you need the following python packages:
* vbench: https://github.com/wesm/vbench
You have to download/clone it from the github page and use
"python setup.py install"

vbench has two more dependencies (those are available on pypi.python.org so 
you can simply use easy_install) 
* sqlalchemy: http://pypi.python.org/pypi/SQLAlchemy/0.7.5
* pandas: http://pypi.python.org/pypi/pandas/0.7.0rc1

vbench currently only supports git repositories. As Brian uses a SVN repository,
it has to use "git svn clone" first to convert it to git -- this should be done
automatically but you obviously need a working installation of git.

Benchmark results are saved in a database, therefore if you add new benchmarks
or if new revisions appear, only the new things will be run. It might make sense
to delete the database and re-run all benchmarks from time to time to avoid 
treating improvements/regressions in external libraries as numpy/scipy as 
improvements /regressions in Brian.
'''

import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt

try:
    from vbench.api import BenchmarkRunner, GitRepo
except ImportError:
    raise ImportError('You need to have vbench installed: https://github.com/wesm/vbench')

import benchmark_connections, benchmark_spikegenerator, benchmark_multiplespikegenerator

# NECESSARY CHANGE HERE: ******************************************************
# Use an absolute path -- if the directory does not exist it will be created
PATH = '/home/marcel/data/vbench'
# *****************************************************************************

SVN_URL = 'https://neuralensemble.org/svn/brian/trunk'
REPO_PATH = os.path.join(PATH, 'brian-trunk')
if not os.path.exists(REPO_PATH):
    # create the repository
    os.makedirs(REPO_PATH)
    os.system('git svn clone %s %s' % (SVN_URL, REPO_PATH))
else:
    # update the repository (can only be called from the directory...) 
    os.chdir(REPO_PATH)
    os.system('git svn fetch')
    
REPO_URL = 'file://' + REPO_PATH
DB_PATH = os.path.join(PATH, 'benchmarks.db')

TMP_DIR = tempfile.mkdtemp(suffix='vbench')
 
# Those two are not really needed at the moment as no C extensions are compiled
# by default 
PREPARE = """
python2 setup.py clean
"""
BUILD = """
python2 setup.py build_ext
"""

START_DATE = datetime(2008, 9, 23) # Brian version 1.0.0

repo = GitRepo(REPO_PATH)

#TODO: Replace this with an automatic search for benchmarks
benchmarks = [benchmark_connections.bench_sparse,
              benchmark_connections.bench_dynamic,
              benchmark_connections.bench_dense,
              benchmark_spikegenerator.bench_pairs,
              benchmark_spikegenerator.bench_array,
              benchmark_spikegenerator.bench_bigarray,
              benchmark_multiplespikegenerator.bench_multiple]

if __name__ == '__main__':
    runner = BenchmarkRunner(benchmarks, REPO_PATH, REPO_URL, BUILD, DB_PATH,
                             TMP_DIR, PREPARE, run_option='eod',
                             start_date=START_DATE)
    runner.run()
    
    # Plot the results
    for benchmark in benchmarks:
        benchmark.plot(DB_PATH)
    
    plt.show()    