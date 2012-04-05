echo Using $(which python)

# if no system-wide packages are used, update numpy etc. to newest version
if [ $packages = newest ]; then
  echo "Using newest available package versions"
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade -I numpy 
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade -I scipy sympy
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade -I matplotlib
elif [ $packages = oldest ]; then
  echo "Using oldest available package versions supported by Brian"
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install -I numpy==1.3.0 
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install http://sourceforge.net/projects/scipy/files/scipy/0.7.0/scipy-0.7.0.tar.gz/download
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install -I sympy
  # Brian depencies state matplotlib>=0.90.1 but 0.98.1 is the oldest version still available
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install -I http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.98.1/matplotlib-0.98.1.tar.gz/download
fi

# Print the version numbers for the dependencies
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import numpy; print 'numpy version: ', numpy.__version__"
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import scipy; print 'scipy version: ', scipy.__version__"
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import sympy; print 'sympy version: ', sympy.__version__"
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import matplotlib; print 'matplotlib version: ', matplotlib.__version__"

# Make sure the build ends up in the build/lib directory
python setup.py build --build-lib=build/lib
