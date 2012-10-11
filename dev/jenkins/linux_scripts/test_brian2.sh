CURRENT=$(pwd)
cd /home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages 
# get the newest version of nose and coverage, ignoring installed packages
bin/pip install --upgrade -I nose coverage || :

# Make sure pyparsing is installed
bin/pip install pyparsing

# This is copied from the build_brian.sh script. When brian2 gets a setup.py script, this script
# should be used and the following lines removed.
##### From build_brian.sh

echo "Using newest available package versions"
bin/pip install --upgrade numpy 
bin/pip install --upgrade scipy
bin/pip install sympy==0.7.1
bin/pip install --upgrade matplotlib    
# Brian depencies state matplotlib>=0.90.1 but 0.98.5.3 seems to be the oldest one installable
bin/pip install http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.98.5/matplotlib-0.98.5.3.tar.gz/download

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
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage run --rcfile=.coveragerc_brian2 ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --with-xunit --logging-clear-handlers --verbose brian2/tests || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage xml --rcfile=.coveragerc_brian2 || :
