@echo off
swig -python -c++ fastexp.i
setup.py build_ext --inplace -c mingw32
