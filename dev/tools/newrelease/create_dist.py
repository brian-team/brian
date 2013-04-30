import os
from distutils.core import run_setup

pathname = os.path.abspath(os.path.dirname(__file__))
os.chdir(pathname)
os.chdir('../../../.') # work from Brian's root
# Create a windows installer for the pure Python version
os.environ['BRIAN_SETUP_NO_EXTENSIONS'] = '1' 
run_setup('setup.py', ['bdist_wininst', '--plat-name=win32'])
del os.environ['BRIAN_SETUP_NO_EXTENSIONS']
run_setup('setup.py', ['sdist', '--formats=gztar,zip'])
