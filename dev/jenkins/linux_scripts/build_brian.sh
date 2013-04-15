#! /bin/bash
echo Using $(which python)

OLD_DIR="$(pwd)"

# change into the virtualenv directory
cd ~/.jenkins/virtual_envs/$PythonVersion/$packages

# if no system-wide packages are used, update numpy etc. to newest version
if [[ $packages == "newest" ]]; then
  echo "Using newest available package versions"
  bin/pip install --upgrade numpy 
  bin/pip install --upgrade scipy
  
  if [[ ${PythonVersion:0:1} == '2' ]]; then
  	bin/pip install --upgrade sympy
  else
  	bin/pip http://sympy.googlecode.com/files/sympy-0.7.2-py3.2.tar.gz
  fi
  if [[ $PythonVersion == "python2.5" ]]; then
    # matplotlib 1.2 is no longer compatible with Python 2.5
    bin/pip install --upgrade matplotlib==1.1
  else
    bin/pip install --upgrade matplotlib
  fi
elif [[ $packages == "oldest" ]]; then
  echo "Using oldest available package versions supported by Brian"
  bin/pip install numpy==1.4.1
  # scipy 0.7 has a bug that makes it impossible to use weave, download the
  # package and apply the patch before installing
  mkdir downloads || :
  wget -O downloads/scipy-0.7.0.tar.gz http://sourceforge.net/projects/scipy/files/scipy/0.7.0/scipy-0.7.0.tar.gz/download
  cd downloads
  tar xvf scipy-0.7.0.tar.gz
  # get and apply patch
  cd scipy-0.7.0
  wget http://projects.scipy.org/scipy/raw-attachment/ticket/739/weave-739.patch
  patch -p1 < weave-739.patch
  # build scipy
  ../../bin/python setup.py install
  cd ../..
  
  bin/pip install sympy
  # Brian depencies state matplotlib>=0.90.1 but 0.98.5.3 seems to be the oldest one installable
  bin/pip install http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.98.5/matplotlib-0.98.5.3.tar.gz/download
fi

# Print the version numbers for the dependencies
bin/python -c "import numpy; print 'numpy version: ', numpy.__version__"
bin/python -c "import scipy; print 'scipy version: ', scipy.__version__"
bin/python -c "import sympy; print 'sympy version: ', sympy.__version__"
bin/python -c "import matplotlib; print 'matplotlib version: ', matplotlib.__version__"

cd "$OLD_DIR"
# Make sure the build ends up in the build/lib directory
python setup.py build --build-lib=build/lib
