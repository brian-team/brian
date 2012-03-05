echo Using $(which python)

# if no system-wide packages are used, update numpy etc. to newest version
if [ $packages = no_global ]; then
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade numpy 
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade scipy sympy
  ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/pip install --upgrade matplotlib
fi

# Print the version numbers for the dependencies
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import numpy; print 'numpy version: ', numpy.__version__"
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import scipy; print 'scipy version: ', scipy.__version__"
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import sympy; print 'sympy version: ', sympy.__version__"
~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/python -c "import matplotlib; print 'matplotlib version: ', matplotlib.__version__"

# Make sure the build ends up in the build/lib directory
python setup.py build --build-lib=build/lib
