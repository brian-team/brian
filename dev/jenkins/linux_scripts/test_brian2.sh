CURRENT=$(pwd)
cd /home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages 
# get the newest version of nose and coverage, ignoring installed packages
bin/pip install --upgrade -I nose coverage || :

# Make sure pyparsing and ipython (used for pretty printing) are installed
bin/pip install pyparsing
bin/pip install ipython

# Make sure we have sphinx (for testing the sphinxext)
bin/pip install sphinx

# This is copied from the build_brian.sh script. When brian2 gets a setup.py script, this script
# should be used and the following lines removed.
##### From build_brian.sh

echo "Using newest available package versions"
bin/pip install --upgrade numpy 
bin/pip install --upgrade scipy
bin/pip install sympy==0.7.1
bin/pip install --upgrade matplotlib

# Print the version numbers for the dependencies
bin/python -c "import numpy; print 'numpy version: ', numpy.__version__"
bin/python -c "import scipy; print 'scipy version: ', scipy.__version__"
bin/python -c "import sympy; print 'sympy version: ', sympy.__version__"
bin/python -c "import matplotlib; print 'matplotlib version: ', matplotlib.__version__"

##### End build_brian.sh

cd "$CURRENT"
# Directly use the source directory (no setup.py yet)
export PYTHONPATH="$(pwd)":$PYTHONPATH

# delete remaining compiled code from previous runs
echo deleting '~/.python*_compiled' if it exists
rm -r ~/.python*_compiled || :

# Run unit tests and record coverage but do not fail the build if anything goes wrong here
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage erase --rcfile=.coveragerc_brian2 || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage run --rcfile=.coveragerc_brian2 ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --with-xunit --logging-clear-handlers --verbose --with-doctest brian2 || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage xml --rcfile=.coveragerc_brian2 || :
