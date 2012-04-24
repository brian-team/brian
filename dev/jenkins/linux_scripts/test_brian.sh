# get the newest version of nose and coverage, ignoring installed packages
/home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade -I nose coverage || :

# Use the previously built brian version
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH

# Run unit tests and record coverage but do not fail the build if anything goes wrong here
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage -e || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage run --branch ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --verbose brian/tests || :
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/coverage xml --include='brian/*' || :
