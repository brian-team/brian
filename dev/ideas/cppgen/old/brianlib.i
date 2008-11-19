%module brianlib

%include "std_string.i"
%include "std_vector.i"

%{
#define SWIG_FILE_WITH_INIT
#include "brianlib.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

// Use this technique to create a vector<double> argument
namespace std {
   %template(DoubleVector) vector<double>;
}

// use this technique to pass numpy arrays to a function with the
// given arguments
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *b, int n)};

%include "brianlib.h"
