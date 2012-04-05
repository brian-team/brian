#! /bin/sh
mkdir -p ~/.jenkins/virtual_envs
# replace this by the list of python versions available
for PYTHON in python2.6 python2.7; do
  for PACKAGES in oldest newest; do 
      virtualenv --python=$PYTHON --no-site-packages ~/.jenkins/virtual_envs/$PYTHON/$PACKAGES
  done
done
