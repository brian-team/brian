%module fastexp

%{
#define SWIG_FILE_WITH_INIT
#include "fastexp.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *x, int n)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *y, int m)};

%include "fastexp.h"
 