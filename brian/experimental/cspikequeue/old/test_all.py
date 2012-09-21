import os, subprocess

SEPARATOR = '/'*20


def run_and_print(command, print_out = True):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if print_out:
        for line in p.stdout.readlines():
            print line,
    retval = p.wait()
    return retval

run_and_print("rm spikequeue")

if True:
    # The c++ test
    CPP_BUILD = 'g++ spikequeue.cpp -O3 -o spikequeue'
    if run_and_print(CPP_BUILD):
        print SEPARATOR+' C++ file built'

if False:
    CPP_RUN= './spikequeue'
    if run_and_print(CPP_RUN):
        print SEPARATOR+' Run successful'

if True:
    run_and_print("rm _cspikequeue.so")
    # the swig test
    SWIG_BUILD = 'swig -c++ -python spikequeue.i'
    run_and_print(SWIG_BUILD)
    print SEPARATOR+' SWIG files built'
    
    PYTHON_BUILD = 'python setup.py build_ext --inplace'
    run_and_print(PYTHON_BUILD)
    print SEPARATOR+'Python files built'

if False:
    PYTHON_TEST = 'python test_python.py'
    run_and_print(PYTHON_TEST)
    print SEPARATOR+' Python test successful'
if True:
    PYTHON_TEST = 'python test_two_synapses.py'
    run_and_print(PYTHON_TEST)
    print SEPARATOR+' Python test successful'
