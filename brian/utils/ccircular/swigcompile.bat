@echo off
swig -python -c++ ccircular.i
setup.py build_ext --inplace -c mingw32
