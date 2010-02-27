@echo off
swig -python -c++ ccircular.i
rem call swigcompile.bat with arguments -D USE_EXPANDING_SPIKECONTAINER
rem to use the new dynamic array based SpikeContainer object
setup.py build_ext --inplace %*
