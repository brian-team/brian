# get the newest version of nose and coverage, ignoring installed packages
/home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade -I nose coverage || :

# Make sure pyparsing is installed
bin/pip install pyparsing

# Directly use the source directory (no setup.py yet)
export PYTHONPATH="$(pwd)":$PYTHONPATH

# delete remaining compiled code from previous runs
echo deleting '~/.python*_compiled' if it exists
rm -r ~/.python*_compiled || :

# Run unit tests and record coverage but do not fail the build if anything goes wrong here
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage erase --rcfile=.coveragerc_brian2 || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage run --rcfile=.coveragerc_brian2 ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --with-xunit --logging-clear-handlers --verbose brian2/tests || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage xml --rcfile=.coveragerc_brian2 || :
