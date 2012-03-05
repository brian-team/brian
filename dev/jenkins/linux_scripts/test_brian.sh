# get the newest version of nose and coverage
/home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade nose coverage || :

# Use the previously built brian version
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH

# Run unit tests and record coverage but do not fail the build if anything goes wrong here
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --with-xunit --with-coverage --cover-html --cover-package=brian --verbose brian/tests || :
