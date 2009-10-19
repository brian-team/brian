@echo off
swig -python -c++ testchagpp.i
setup.py build_ext --inplace -c mingw32
