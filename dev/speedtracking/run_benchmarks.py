'''
Run benchmarks for all revisions of Brian in the repository, starting with
version 1.0.0 and only using every tenth revision.
To run the benchmark on your machine you need the following python packages:
* vbench: https://github.com/wesm/vbench
You have to download/clone it from the github page and use
"python setup.py install"

vbench has two more dependencies (those are available on pypi.python.org so 
you can simply use easy_install or pip) 
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

import os, sys
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt

try:
    from vbench.api import Benchmark, BenchmarkRunner, GitRepo
except ImportError:
    raise ImportError('You need to have vbench installed: https://github.com/wesm/vbench')

# inspired by https://github.com/wesm/pandas/blob/master/vb_suite/suite.py
modules = ['benchmark_connections',
           'benchmark_spikegenerator',
           'benchmark_multiplespikegenerator',
           'benchmark_stdp']

by_module = {}
benchmarks = []

# automatically adds all the benchmarks from the above modules
for modname in modules:
    ref = __import__(modname)
    by_module[modname] = [v for v in ref.__dict__.values()
                          if isinstance(v, Benchmark)]
    benchmarks.extend(by_module[modname])

if __name__ == '__main__':
    if len(sys.argv) != 2 or not os.path.isabs(sys.argv[1]):
        sys.stderr.write('Usage: python run_benchmarks.py PATH\n' +
                         'where PATH is an absolute path that will be used as the base path for saving results.\n')
        sys.exit(1)

    # if the directory does not exist it will be created
    PATH = sys.argv[1]
    
    DB_PATH = os.path.join(PATH, 'benchmarks.db')
    
    SVN_URL = 'https://neuralensemble.org/svn/brian/trunk'
    REPO_PATH = os.path.join(PATH, 'brian-trunk')
    if not os.path.exists(REPO_PATH):
        # create the repository
        os.makedirs(REPO_PATH)
        os.system('git svn clone %s %s' % (SVN_URL, REPO_PATH))
    else:
        # update the repository (can only be called from the directory...) 
        os.chdir(REPO_PATH)
        os.system('git svn rebase')
        os.system('git svn fetch')

    REPO_URL = 'file://' + REPO_PATH

    TMP_DIR = tempfile.mkdtemp(suffix='vbench')

    # Those two are not really needed at the moment as no C extensions are
    # compiled by default
    # TODO: Does using sys.executable here work on Windows? 
    PREPARE = "%s setup.py clean" % sys.executable
    BUILD = "%s setup.py build_ext" % sys.executable

    START_DATE = datetime(2008, 9, 23) # Brian version 1.0.0

    repo = GitRepo(REPO_PATH)

    runner = BenchmarkRunner(benchmarks, REPO_PATH, REPO_URL, BUILD, DB_PATH,
                             TMP_DIR, PREPARE, run_option=10, #every 10 revs
                             start_date=START_DATE)
    runner.run()

    # Plot the results
    for benchmark in benchmarks:
        benchmark.plot(DB_PATH)

    plt.show()
