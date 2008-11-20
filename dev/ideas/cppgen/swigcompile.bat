@echo off
swig -python -c++ brianlib.i
setup.py build_ext --inplace -c mingw32
